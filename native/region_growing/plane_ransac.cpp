#include "plane_ransac.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_set>

namespace {

constexpr float kDegToRad = static_cast<float>(PI / 180.0);

bool fit_plane_from_indices(
    const PointCloud& pcd,
    const std::vector<size_t>& indices,
    PlaneModel& plane
){
    if (indices.size() < 3){
        return false;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 3> pts(indices.size(), 3);
    Point centroid = Point{0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < indices.size(); ++i){
        const Point& p = pcd.coords[indices[i]];
        pts.row(i) = p.transpose();
        centroid += p;
    }
    centroid /= static_cast<float>(indices.size());
    pts.rowwise() -= centroid.transpose();
    const Eigen::Matrix3f covariance = pts.transpose() * pts;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success){
        return false;
    }
    Point normal = solver.eigenvectors().col(0);
    const float n_norm = normal.norm();
    if (n_norm < 1e-8f){
        return false;
    }
    normal /= n_norm;
    plane.centroid = centroid;
    plane.normal = normal;
    plane.d = -plane.normal.dot(plane.centroid);
    plane.indices = indices;
    return true;
}

bool fit_plane_from_sample(
    const PointCloud& pcd,
    size_t a,
    size_t b,
    size_t c,
    float cos_threshold,
    PlaneModel& plane
){
    const Point& p0 = pcd.coords[a];
    const Point& p1 = pcd.coords[b];
    const Point& p2 = pcd.coords[c];
    Point normal = (p1 - p0).cross(p2 - p0);
    const float n_norm = normal.norm();
    if (n_norm < 1e-8f){
        return false;
    }
    normal /= n_norm;
    if (!pcd.normals.empty()){
        const float d0 = std::abs(pcd.normals[a].dot(normal));
        const float d1 = std::abs(pcd.normals[b].dot(normal));
        const float d2 = std::abs(pcd.normals[c].dot(normal));
        if (d0 < cos_threshold || d1 < cos_threshold || d2 < cos_threshold){
            return false;
        }
    }
    plane.normal = normal;
    plane.centroid = (p0 + p1 + p2) / 3.0f;
    plane.d = -plane.normal.dot(plane.centroid);
    plane.indices.clear();
    return true;
}

inline float point_plane_distance(const Point& p, const PlaneModel& plane){
    return std::abs(plane.normal.dot(p) + plane.d);
}

std::vector<size_t> extract_largest_component(
    const PointCloud& pcd,
    PlaneRansacKDTree& kdtree,
    const std::vector<size_t>& inliers,
    float cluster_epsilon
){
    if (inliers.empty()){
        return {};
    }

    std::unordered_set<size_t> inlier_set;
    inlier_set.reserve(inliers.size() * 2);
    for (const auto idx : inliers){
        inlier_set.insert(idx);
    }

    std::unordered_set<size_t> visited;
    visited.reserve(inliers.size() * 2);
    std::vector<size_t> largest;

    for (const auto seed : inliers){
        if (visited.find(seed) != visited.end()){
            continue;
        }
        std::vector<size_t> component;
        std::queue<size_t> q;
        q.push(seed);
        visited.insert(seed);

        while (!q.empty()){
            const size_t cur = q.front();
            q.pop();
            component.push_back(cur);

            PlaneRansacIndicesDist indices_dist;
            PlaneRansacRadiusSet result_set(cluster_epsilon, indices_dist);
            kdtree.findNeighbors(
                result_set,
                pcd.coords[cur].data(),
                nanoflann::SearchParameters(0.0f, false)
            );
            for (const auto& item : indices_dist){
                const size_t nbr = item.first;
                if (inlier_set.find(nbr) == inlier_set.end()){
                    continue;
                }
                if (visited.insert(nbr).second){
                    q.push(nbr);
                }
            }
        }

        if (component.size() > largest.size()){
            largest = std::move(component);
        }
    }

    return largest;
}

size_t update_required_iterations(
    size_t best_support,
    size_t active_size,
    float probability
){
    if (best_support < 3 || active_size < 3 || best_support > active_size){
        return std::numeric_limits<size_t>::max();
    }
    const double w = static_cast<double>(best_support) / static_cast<double>(active_size);
    const double good_sample_prob = w * w * w;
    if (good_sample_prob <= 0.0){
        return std::numeric_limits<size_t>::max();
    }
    if (good_sample_prob >= 1.0){
        return 1;
    }
    const double miss_prob = std::clamp(static_cast<double>(probability), 1e-9, 0.999999);
    const double denom = std::log1p(-good_sample_prob);
    if (!std::isfinite(denom) || denom >= 0.0){
        return std::numeric_limits<size_t>::max();
    }
    const double numer = std::log(miss_prob);
    const double required = numer / denom;
    if (!std::isfinite(required) || required < 1.0){
        return 1;
    }
    return static_cast<size_t>(std::ceil(required));
}

} // namespace

void compute_plane_ransac_normals(
    const std::vector<Point>& coords,
    std::vector<Point>& normals,
    PlaneRansacKDTree& kdtree,
    float search_radius
){
    normals.resize(coords.size(), Point{0.0f, 0.0f, 1.0f});
    for (size_t i = 0; i < coords.size(); ++i){
        PlaneRansacIndicesDist indices_dist;
        PlaneRansacRadiusSet result_set(search_radius, indices_dist);
        kdtree.findNeighbors(
            result_set,
            coords[i].data(),
            nanoflann::SearchParameters(0.0f, false)
        );
        if (indices_dist.size() < 3){
            continue;
        }

        Eigen::Matrix<float, Eigen::Dynamic, 3> neighbors(indices_dist.size(), 3);
        for (size_t k = 0; k < indices_dist.size(); ++k){
            neighbors.row(k) = (coords[indices_dist[k].first] - coords[i]).transpose();
        }
        const Eigen::Matrix3f covariance = neighbors.transpose() * neighbors;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        if (solver.info() != Eigen::Success){
            continue;
        }
        Point normal = solver.eigenvectors().col(0);
        const float norm = normal.norm();
        if (norm < 1e-8f){
            continue;
        }
        normals[i] = normal / norm;
    }
}

PlaneRansacResult segment_planes_ransac(
    PointCloud& pcd,
    PlaneRansacKDTree& kdtree,
    const PlaneRansacParams& params,
    const PlaneRansacProgressCallback& progress_cb
){
    PlaneRansacResult result;
    const size_t n_points = pcd.coords.size();
    result.labels.assign(n_points, -1);

    std::vector<size_t> active_indices(n_points);
    std::iota(active_indices.begin(), active_indices.end(), 0);

    std::mt19937 rng(params.seed);
    const float epsilon = std::max(1e-6f, params.epsilon);
    const float cluster_epsilon =
        (params.cluster_epsilon > 0.0f) ? params.cluster_epsilon : epsilon;
    const float cos_threshold = std::cos(params.normal_threshold_deg * kDegToRad);
    size_t assigned_points = 0;

    auto emit_progress = [&](const std::string& stage,
                             size_t current_best_support,
                             bool done) -> bool {
        if (!progress_cb){
            return true;
        }
        PlaneRansacRuntime rt;
        rt.stage = stage;
        rt.planes_found = result.planes.size();
        rt.active_points_remaining = active_indices.size();
        rt.current_best_support = current_best_support;
        rt.assigned_points = assigned_points;
        rt.done = done;
        return progress_cb(rt);
    };

    if (!emit_progress("Searching plane 1...", 0, false)){
        result.cancelled = true;
        return result;
    }

    while (active_indices.size() >= params.min_points){
        PlaneModel best_plane;
        std::vector<size_t> best_support;
        size_t required_iterations = params.max_iterations_per_plane;
        size_t iter = 0;
        const size_t active_size = active_indices.size();

        while (iter < params.max_iterations_per_plane &&
               (iter < required_iterations || best_support.size() < params.min_points)){
            ++iter;

            std::uniform_int_distribution<size_t> dist(0, active_indices.size() - 1);
            const size_t ia = active_indices[dist(rng)];
            const size_t ib = active_indices[dist(rng)];
            const size_t ic = active_indices[dist(rng)];
            if (ia == ib || ia == ic || ib == ic){
                continue;
            }

            PlaneModel candidate;
            if (!fit_plane_from_sample(pcd, ia, ib, ic, cos_threshold, candidate)){
                continue;
            }

            std::vector<size_t> support;
            support.reserve(active_indices.size() / 4);
            for (const auto idx : active_indices){
                if (point_plane_distance(pcd.coords[idx], candidate) > epsilon){
                    continue;
                }
                if (!pcd.normals.empty() &&
                    std::abs(pcd.normals[idx].dot(candidate.normal)) < cos_threshold){
                    continue;
                }
                support.push_back(idx);
            }

            if (support.size() > best_support.size()){
                best_support = std::move(support);
                best_plane = candidate;
                required_iterations = std::min(
                    params.max_iterations_per_plane,
                    update_required_iterations(best_support.size(), active_size, params.probability)
                );
                if (!emit_progress(
                        "Searching plane " + std::to_string(result.planes.size() + 1) + "...",
                        best_support.size(),
                        false
                    )){
                    result.cancelled = true;
                    return result;
                }
            } else if ((iter % 100) == 0) {
                if (!emit_progress(
                        "Searching plane " + std::to_string(result.planes.size() + 1) + "...",
                        best_support.size(),
                        false
                    )){
                    result.cancelled = true;
                    return result;
                }
            }
        }

        if (best_support.size() < params.min_points){
            break;
        }

        const std::vector<size_t> largest_component =
            extract_largest_component(pcd, kdtree, best_support, cluster_epsilon);
        if (largest_component.size() < params.min_points){
            break;
        }

        PlaneModel accepted_plane;
        if (!fit_plane_from_indices(pcd, largest_component, accepted_plane)){
            break;
        }

        const int32_t plane_id = static_cast<int32_t>(result.planes.size());
        std::unordered_set<size_t> accepted_set;
        accepted_set.reserve(largest_component.size() * 2);
        for (const auto idx : largest_component){
            accepted_set.insert(idx);
            result.labels[idx] = plane_id;
        }
        assigned_points += largest_component.size();

        std::vector<size_t> remaining_active;
        remaining_active.reserve(active_indices.size() - largest_component.size());
        for (const auto idx : active_indices){
            if (accepted_set.find(idx) == accepted_set.end()){
                remaining_active.push_back(idx);
            }
        }
        active_indices = std::move(remaining_active);
        result.planes.push_back(std::move(accepted_plane));

        if (!emit_progress(
                "planes=" + std::to_string(result.planes.size()) +
                " | remaining=" + std::to_string(active_indices.size()),
                0,
                false
            )){
            result.cancelled = true;
            return result;
        }
    }

    emit_progress(
        "Done",
        0,
        true
    );
    return result;
}
