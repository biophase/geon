#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "connected_components.h"
#include "rgrow.h"
#include "types.h"

void computeNormals(
    const std::vector<Point>& coords,
    std::vector<Point>& normals,
    const std::unordered_set<size_t>& task_inds,
    KDTreeType& kdtree,
    float search_radius
){
    for (const size_t& i : task_inds){
        IndicesDistType indices_dist;
        RadiusSetType result_set(search_radius, indices_dist);
        kdtree.findNeighbors(
            result_set,
            coords.at(i).data(),
            nanoflann::SearchParameters(.0f, false)
        );

        if (indices_dist.size() < 3){
            normals[i] = Point{0.0f, 0.0f, 1.0f};
            continue;
        }

        Eigen::Matrix<float, Eigen::Dynamic, 3> neighbors(indices_dist.size(), 3);
        for (size_t k = 0; k < indices_dist.size(); ++k){
            neighbors.row(k) = coords[indices_dist[k].first] - coords[i];
        }

        Eigen::Matrix3f covariance = neighbors.transpose() * neighbors;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        normals[i] = solver.eigenvectors().col(0);
    }
}

template <typename T>
T randomPop(std::unordered_set<T>& input){
    if (input.empty()){
        throw std::range_error("Set is empty");
    }
    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, input.size() - 1);
    auto it = input.begin();
    std::advance(it, static_cast<long>(dist(rng)));
    T value = *it;
    input.erase(it);
    return value;
}

static inline int clampIndex(int value, int lo, int hi){
    return std::max(lo, std::min(hi, value));
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> getPcdAabb(
    const PointCloud& pcd,
    std::unordered_set<size_t>& task_inds
){
    Eigen::Vector3f min_values(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    );
    Eigen::Vector3f max_values(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    );

    for (const size_t& ind : task_inds){
        min_values = min_values.cwiseMin(pcd.coords[ind]);
        max_values = max_values.cwiseMax(pcd.coords[ind]);
    }

    min_values -= Eigen::Vector3f::Constant(1e-3f);
    max_values += Eigen::Vector3f::Constant(1e-3f);
    return std::pair<Eigen::Vector3f, Eigen::Vector3f>(min_values, max_values);
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> getPcdAabb(const PointCloud& pcd){
    Eigen::Vector3f min_values(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    );
    Eigen::Vector3f max_values(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    );

    for (const auto& xyz : pcd.coords){
        min_values = min_values.cwiseMin(xyz);
        max_values = max_values.cwiseMax(xyz);
    }

    min_values -= Eigen::Vector3f::Constant(1e-3f);
    max_values += Eigen::Vector3f::Constant(1e-3f);
    return std::pair<Eigen::Vector3f, Eigen::Vector3f>(min_values, max_values);
}

std::vector<size_t> unflattenIndex(size_t flat_idx, size_t X, size_t Y, size_t Z){
    std::vector<size_t> c(3);
    c[2] = flat_idx % Z;
    c[1] = (flat_idx / Z) % Y;
    c[0] = flat_idx / (Y * Z);
    return c;
}

size_t flattenIndex(size_t x, size_t X, size_t y, size_t Y, size_t z, size_t Z){
    return x * Y * Z + y * Z + z;
}

subdividePointCloudFixedChunkSize_returnType subdividePointCloudFixedChunkSize(
    const PointCloud& pcd,
    std::unordered_set<size_t> task_inds,
    float chunk_size_x,
    float chunk_size_y,
    float chunk_size_z
){
    const auto [aabb_min_values, aabb_max_values] = getPcdAabb(pcd, task_inds);
    std::unordered_map<size_t, std::unordered_set<size_t>> chunks;
    Eigen::Vector3f aabb_size = aabb_max_values - aabb_min_values;

    const size_t num_chunks_x = static_cast<size_t>(std::ceil(aabb_size[0] / chunk_size_x));
    const size_t num_chunks_y = static_cast<size_t>(std::ceil(aabb_size[1] / chunk_size_y));
    const size_t num_chunks_z = static_cast<size_t>(std::ceil(aabb_size[2] / chunk_size_z));
    const Eigen::Vector3f chunk_size = {chunk_size_x, chunk_size_y, chunk_size_z};

    for (size_t i : task_inds){
        Eigen::Vector3i chunk_coord = (pcd.coords[i] - aabb_min_values).cwiseQuotient(chunk_size).cast<int>();
        chunk_coord[0] = clampIndex(chunk_coord[0], 0, static_cast<int>(num_chunks_x) - 1);
        chunk_coord[1] = clampIndex(chunk_coord[1], 0, static_cast<int>(num_chunks_y) - 1);
        chunk_coord[2] = clampIndex(chunk_coord[2], 0, static_cast<int>(num_chunks_z) - 1);
        size_t coord_hash = flattenIndex(
            static_cast<size_t>(chunk_coord[0]),
            num_chunks_x,
            static_cast<size_t>(chunk_coord[1]),
            num_chunks_y,
            static_cast<size_t>(chunk_coord[2]),
            num_chunks_z
        );
        chunks.emplace(coord_hash, std::unordered_set<size_t>{}).first->second.insert(i);
    }
    return subdividePointCloudFixedChunkSize_returnType{
        chunks,
        num_chunks_x,
        num_chunks_y,
        num_chunks_z
    };
}

std::unordered_map<size_t, std::unordered_set<size_t>> subdividePointCloud(
    const PointCloud& pcd,
    size_t num_chunks_x,
    size_t num_chunks_y,
    size_t num_chunks_z
){
    const auto [aabb_min_values, aabb_max_values] = getPcdAabb(pcd);
    std::unordered_map<size_t, std::unordered_set<size_t>> chunks;
    Eigen::Vector3f aabb_size = aabb_max_values - aabb_min_values;
    Eigen::Vector3f chunk_size = {
        aabb_size[0] / static_cast<float>(num_chunks_x),
        aabb_size[1] / static_cast<float>(num_chunks_y),
        aabb_size[2] / static_cast<float>(num_chunks_z),
    };

    for (size_t i = 0; i < pcd.coords.size(); ++i){
        Eigen::Vector3i chunk_coord = (pcd.coords[i] - aabb_min_values).cwiseQuotient(chunk_size).cast<int>();
        chunk_coord[0] = clampIndex(chunk_coord[0], 0, static_cast<int>(num_chunks_x) - 1);
        chunk_coord[1] = clampIndex(chunk_coord[1], 0, static_cast<int>(num_chunks_y) - 1);
        chunk_coord[2] = clampIndex(chunk_coord[2], 0, static_cast<int>(num_chunks_z) - 1);
        size_t coord_hash = flattenIndex(
            static_cast<size_t>(chunk_coord[0]),
            num_chunks_x,
            static_cast<size_t>(chunk_coord[1]),
            num_chunks_y,
            static_cast<size_t>(chunk_coord[2]),
            num_chunks_z
        );
        chunks.emplace(coord_hash, std::unordered_set<size_t>{}).first->second.insert(i);
    }
    return chunks;
}

static void refitPlane(Region& reg, const PointCloud& pcd){
    if (reg.indices.empty()){
        reg.centroid = Point{0.0f, 0.0f, 0.0f};
        reg.normal = Point{0.0f, 0.0f, 1.0f};
        return;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 3> reg_coords(reg.indices.size(), 3);
    Point centroid = {0.0f, 0.0f, 0.0f};
    size_t idx_out = 0;
    for (const auto& ind : reg.indices){
        reg_coords.row(idx_out) = pcd.coords[ind];
        centroid += pcd.coords[ind];
        ++idx_out;
    }

    centroid /= static_cast<float>(reg.indices.size());
    reg_coords.rowwise() -= centroid.transpose();
    Eigen::Matrix3f covariance = reg_coords.transpose() * reg_coords;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    reg.normal = solver.eigenvectors().col(0);
    reg.centroid = centroid;
}

static size_t computeFailedAttemptLimit(size_t point_count, size_t tau, float confidence){
    if (point_count == 0 || tau == 0){
        return std::numeric_limits<size_t>::max();
    }

    const size_t g = std::min(tau, point_count);
    const double p_hit_raw = static_cast<double>(g) / static_cast<double>(point_count);
    const double p_hit = std::clamp(p_hit_raw, 1e-9, 1.0 - 1e-9);
    const double conf = std::clamp(static_cast<double>(confidence), 1e-6, 0.999999);
    const double denom = std::log1p(-p_hit);
    const double numer = std::log1p(-conf);
    const double attempts = numer / denom;
    return std::max<size_t>(1, static_cast<size_t>(std::ceil(attempts)));
}

static bool isPlanarSeed(
    size_t seed_idx,
    const PointCloud& pcd,
    KDTreeType& kdtree,
    const RegionGrowingParams& p
){
    if (!p.enable_seed_gating){
        return true;
    }
    IndicesDistType indices_dist;
    RadiusSetType result_set(p.epsilon, indices_dist);
    kdtree.findNeighbors(
        result_set,
        pcd.coords[seed_idx].data(),
        nanoflann::SearchParameters(p.search_radius_approx, false)
    );
    if (indices_dist.size() < p.seed_min_neighbors){
        return false;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 3> neighbors(indices_dist.size(), 3);
    for (size_t k = 0; k < indices_dist.size(); ++k){
        neighbors.row(k) = pcd.coords[indices_dist[k].first] - pcd.coords[seed_idx];
    }
    const Eigen::Matrix3f covariance = neighbors.transpose() * neighbors;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    const Eigen::Vector3f evals = solver.eigenvalues();
    const float l1 = std::max(0.0f, evals[0]);
    const float l2 = std::max(0.0f, evals[1]);
    const float l3 = std::max(1e-8f, evals[2]);
    const float planarity = (l2 - l1) / l3;
    const float scattering = l1 / l3;
    return (planarity >= p.seed_planarity_min) && (scattering <= p.seed_scattering_max);
}

static bool isUsefulRegion(const Region& reg, const PointCloud& pcd, const RegionGrowingParams& p){
    if (reg.indices.size() < p.min_points_in_region){
        return false;
    }
    const float eps_safe = std::max(1e-8f, p.epsilon);
    float sum_abs_residual = 0.0f;
    for (const auto idx : reg.indices){
        sum_abs_residual += std::abs((pcd.coords[idx] - reg.centroid).dot(reg.normal));
    }
    const float mean_residual = sum_abs_residual / static_cast<float>(reg.indices.size());
    return (mean_residual / eps_safe) <= p.epsilon_multiplier_average;
}

Region growRegionFromSeed(
    PointCloud& pcd,
    KDTreeType& kdtree,
    size_t seed_idx,
    const std::unordered_set<size_t>* task_indices,
    RegionGrowingParams params
){
    auto& p = params;
    Region empty_region;
    if (seed_idx >= pcd.coords.size()){
        return empty_region;
    }

    const float cos_alpha = std::cos(p.alpha);
    std::unordered_set<size_t> available_inds;
    if (task_indices == nullptr){
        for (size_t i = 0; i < pcd.coords.size(); ++i){
            available_inds.insert(i);
        }
    } else {
        available_inds = *task_indices;
    }

    if (available_inds.find(seed_idx) == available_inds.end()){
        return empty_region;
    }
    if (!isPlanarSeed(seed_idx, pcd, kdtree, p)){
        return empty_region;
    }

    Region reg;
    std::unordered_set<size_t> front_inds;
    std::unordered_set<size_t> cand_inds_buffer;
    size_t next_refit = p.first_refit;

    reg.indices.insert(seed_idx);
    reg.centroid = pcd.coords[seed_idx];
    reg.normal = pcd.normals[seed_idx];
    front_inds.insert(seed_idx);
    available_inds.erase(seed_idx);

    while (!front_inds.empty()){
        std::unordered_set<size_t> front_inds_new;
        cand_inds_buffer.clear();

        for (const size_t& f_i : front_inds){
            IndicesDistType indices_dist;
            RadiusSetType result_set(p.epsilon, indices_dist);
            kdtree.findNeighbors(
                result_set,
                pcd.coords[f_i].data(),
                nanoflann::SearchParameters(p.search_radius_approx, false)
            );
            for (const auto& i_d : indices_dist){
                cand_inds_buffer.insert(i_d.first);
            }
        }

        for (const size_t& i : cand_inds_buffer){
            if (available_inds.find(i) == available_inds.end()){
                continue;
            }
            const float reg_normal_dot_fnormal = pcd.normals[i].dot(reg.normal);
            const Point coord_local = pcd.coords[i] - reg.centroid;
            const float dist_to_plane = std::abs(coord_local.dot(reg.normal));
            const float coord_norm = coord_local.norm();
            const float dist_ratio = std::clamp(coord_norm / p.max_dist_from_cent, 0.0f, 1.0f);
            const float cos_thresh = cos_alpha + (1.0f - cos_alpha) * dist_ratio;
            const bool angle_ok = (
                (!p.oriented_normals && std::abs(reg_normal_dot_fnormal) > cos_thresh) ||
                (p.oriented_normals && reg_normal_dot_fnormal > cos_thresh)
            );
            if (dist_to_plane <= p.epsilon_multiplier * p.epsilon && angle_ok){
                front_inds_new.insert(i);
                reg.indices.insert(i);
                available_inds.erase(i);
            }
        }

        front_inds = std::move(front_inds_new);

        if (reg.indices.size() >= next_refit){
            const size_t proposed_next = static_cast<size_t>(std::ceil(
                static_cast<double>(next_refit) * static_cast<double>(p.refit_multiplier)
            ));
            next_refit = std::max(next_refit + static_cast<size_t>(1), proposed_next);
            refitPlane(reg, pcd);

            for (auto it = reg.indices.begin(); it != reg.indices.end();){
                if (*it == seed_idx){
                    ++it;
                    continue;
                }
                const float reg_normal_dot_pnormal = pcd.normals[*it].dot(reg.normal);
                const Point coord_local = pcd.coords[*it] - reg.centroid;
                const float dist_to_plane = std::abs(coord_local.dot(reg.normal));
                const float coord_norm = coord_local.norm();
                const float dist_ratio = std::clamp(coord_norm / p.max_dist_from_cent, 0.0f, 1.0f);
                const float cos_thresh = cos_alpha + (1.0f - cos_alpha) * dist_ratio;
                const bool angle_ok = (
                    (!p.oriented_normals && std::abs(reg_normal_dot_pnormal) > cos_thresh) ||
                    (p.oriented_normals && reg_normal_dot_pnormal > cos_thresh)
                );

                if (dist_to_plane > p.epsilon_multiplier * p.epsilon || !angle_ok){
                    available_inds.insert(*it);
                    it = reg.indices.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    if (reg.indices.empty()){
        return empty_region;
    }

    if (p.perform_cca){
        auto cca_results = unionFindCCA(pcd, reg.indices, p.epsilon);
        std::unordered_set<size_t> chosen;
        for (const auto& comp_inds : cca_results){
            if (comp_inds.find(seed_idx) != comp_inds.end()){
                chosen = comp_inds;
                break;
            }
        }
        if (!chosen.empty()){
            reg.indices = std::move(chosen);
        }
    }

    refitPlane(reg, pcd);
    if (reg.indices.size() < p.min_points_in_region){
        reg.indices.clear();
    }
    return reg;
}

regionGrowing_returnType regionGrowing(
    PointCloud& pcd,
    KDTreeType& kdtree,
    const std::unordered_set<size_t>* task_indices,
    RegionGrowingParams params,
    const std::function<void(const RegionGrowingRuntime&)>& progress_cb
){
    auto& p = params;
    const float cos_alpha = std::cos(p.alpha);
    std::unordered_set<size_t> unassigned_inds;
    std::vector<Region> regions;

    if (task_indices == nullptr){
        for (size_t i = 0; i < pcd.coords.size(); ++i){
            unassigned_inds.insert(i);
        }
    } else {
        unassigned_inds = *task_indices;
    }

    size_t failed_attempts = 0;
    size_t failed_attempt_limit = 0;
    size_t attempts = 0;
    const size_t fail_window = std::max<size_t>(1, p.failrate_window);
    std::vector<uint8_t> fail_hist(fail_window, 0);
    size_t fail_hist_count = 0;
    size_t fail_hist_pos = 0;
    size_t fail_hist_sum = 0;

    auto rolling_fail_rate = [&]() -> float {
        if (fail_hist_count == 0){
            return 0.0f;
        }
        return static_cast<float>(fail_hist_sum) / static_cast<float>(fail_hist_count);
    };

    auto update_rolling_history = [&](bool attempt_failed){
        const uint8_t val = attempt_failed ? 1 : 0;
        if (fail_hist_count < fail_window){
            fail_hist[fail_hist_count] = val;
            fail_hist_sum += static_cast<size_t>(val);
            ++fail_hist_count;
        } else {
            fail_hist_sum -= static_cast<size_t>(fail_hist[fail_hist_pos]);
            fail_hist[fail_hist_pos] = val;
            fail_hist_sum += static_cast<size_t>(val);
            fail_hist_pos = (fail_hist_pos + 1) % fail_window;
        }
    };

    auto emit_runtime = [&](bool done){
        if (!progress_cb){
            return;
        }
        RegionGrowingRuntime rt;
        rt.attempts = attempts;
        rt.failed_attempts = failed_attempts;
        rt.failed_attempt_limit = failed_attempt_limit;
        rt.rolling_fail_rate = rolling_fail_rate();
        rt.rolling_fail_threshold = p.failrate_threshold;
        rt.unassigned_remaining = unassigned_inds.size();
        rt.regions_found = regions.size();
        rt.done = done;
        progress_cb(rt);
    };

    while (!unassigned_inds.empty()){
        if (fail_hist_count >= fail_window && rolling_fail_rate() >= p.failrate_threshold){
            break;
        }
        const size_t seed_idx = randomPop<size_t>(unassigned_inds);
        ++attempts;
        if (!isPlanarSeed(seed_idx, pcd, kdtree, p)){
            ++failed_attempts;
            update_rolling_history(true);
            emit_runtime(false);
            continue;
        }
        Region reg;
        std::unordered_set<size_t> front_inds;
        std::unordered_set<size_t> cand_inds_buffer;
        size_t next_refit = p.first_refit;
        bool region_success = false;
        bool useful_region_success = false;

        reg.indices.insert(seed_idx);
        reg.centroid = pcd.coords[seed_idx];
        reg.normal = pcd.normals[seed_idx];
        front_inds.insert(seed_idx);

        while (!front_inds.empty()){
            std::unordered_set<size_t> front_inds_new;
            cand_inds_buffer.clear();

            for (const size_t& f_i : front_inds){
                IndicesDistType indices_dist;
                RadiusSetType result_set(p.epsilon, indices_dist);
                kdtree.findNeighbors(
                    result_set,
                    pcd.coords[f_i].data(),
                    nanoflann::SearchParameters(p.search_radius_approx, false)
                );
                for (const auto& i_d : indices_dist){
                    cand_inds_buffer.insert(i_d.first);
                }
            }

            for (const size_t& i : cand_inds_buffer){
                if (unassigned_inds.find(i) == unassigned_inds.end()){
                    continue;
                }
                const float reg_normal_dot_fnormal = pcd.normals[i].dot(reg.normal);
                const Point coord_local = pcd.coords[i] - reg.centroid;
                const float dist_to_plane = std::abs(coord_local.dot(reg.normal));
                const float coord_norm = coord_local.norm();
                const float dist_ratio = std::clamp(coord_norm / p.max_dist_from_cent, 0.0f, 1.0f);
                const float cos_thresh = cos_alpha + (1.0f - cos_alpha) * dist_ratio;
                const bool angle_ok = (
                    (!p.oriented_normals && std::abs(reg_normal_dot_fnormal) > cos_thresh) ||
                    (p.oriented_normals && reg_normal_dot_fnormal > cos_thresh)
                );

                if (dist_to_plane <= p.epsilon_multiplier * p.epsilon && angle_ok){
                    front_inds_new.insert(i);
                    reg.indices.insert(i);
                    unassigned_inds.erase(i);
                }
            }

            front_inds = std::move(front_inds_new);

            if (reg.indices.size() >= next_refit){
                const size_t proposed_next = static_cast<size_t>(std::ceil(
                    static_cast<double>(next_refit) * static_cast<double>(p.refit_multiplier)
                ));
                next_refit = std::max(next_refit + static_cast<size_t>(1), proposed_next);
                refitPlane(reg, pcd);

                for (auto it = reg.indices.begin(); it != reg.indices.end();){
                    const float reg_normal_dot_pnormal = pcd.normals[*it].dot(reg.normal);
                    const Point coord_local = pcd.coords[*it] - reg.centroid;
                    const float dist_to_plane = std::abs(coord_local.dot(reg.normal));
                    const float coord_norm = coord_local.norm();
                    const float dist_ratio = std::clamp(coord_norm / p.max_dist_from_cent, 0.0f, 1.0f);
                    const float cos_thresh = cos_alpha + (1.0f - cos_alpha) * dist_ratio;
                    const bool angle_ok = (
                        (!p.oriented_normals && std::abs(reg_normal_dot_pnormal) > cos_thresh) ||
                        (p.oriented_normals && reg_normal_dot_pnormal > cos_thresh)
                    );

                    if (dist_to_plane > p.epsilon_multiplier * p.epsilon || !angle_ok){
                        unassigned_inds.insert(*it);
                        it = reg.indices.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        }

        if (reg.indices.size() < p.min_points_in_region){
            for (const auto& idx : reg.indices){
                unassigned_inds.insert(idx);
            }
        } else if (p.perform_cca){
            auto cca_results = unionFindCCA(pcd, reg.indices, p.epsilon);
            for (const auto& reg_inds : cca_results){
                if (reg_inds.size() >= p.min_points_in_region){
                    Region sub_reg;
                    sub_reg.indices = reg_inds;
                    refitPlane(sub_reg, pcd);
                    useful_region_success = useful_region_success || isUsefulRegion(sub_reg, pcd, p);
                    regions.push_back(std::move(sub_reg));
                    region_success = true;
                } else {
                    for (const auto& idx : reg_inds){
                        unassigned_inds.insert(idx);
                    }
                }
            }
        } else {
            useful_region_success = isUsefulRegion(reg, pcd, p);
            regions.push_back(std::move(reg));
            region_success = true;
        }

        const bool attempt_failed = !useful_region_success;
        if (attempt_failed){
            ++failed_attempts;
        } else {
            failed_attempts = 0;
        }
        update_rolling_history(attempt_failed);
        emit_runtime(false);

        if (p.verbose){
            std::cout
                << "Region attempts: failed=" << failed_attempts
                << "\trolling_fail_rate=" << rolling_fail_rate()
                << "/" << p.failrate_threshold
                << "\tregions=" << regions.size()
                << "\tremaining=" << unassigned_inds.size()
                << std::endl;
        }
    }
    emit_runtime(true);

    std::unordered_map<size_t, size_t> pcd_to_reg_idxmap;
    for (size_t reg_i = 0; reg_i < regions.size(); ++reg_i){
        for (const size_t& idx : regions[reg_i].indices){
            pcd_to_reg_idxmap[idx] = reg_i;
        }
    }

    if (!unassigned_inds.empty() && !regions.empty()){
        auto components = unionFindCCA(pcd, unassigned_inds, p.epsilon);
        std::unordered_set<size_t> changed_regions;

        for (const auto& comp_inds : components){
            if (comp_inds.empty() || comp_inds.size() >= p.min_points_in_region){
                continue;
            }

            std::unordered_set<size_t> candidate_regions;
            for (const auto& idx : comp_inds){
                IndicesDistType indices_dist;
                RadiusSetType result_set(p.epsilon, indices_dist);
                kdtree.findNeighbors(
                    result_set,
                    pcd.coords[idx].data(),
                    nanoflann::SearchParameters(p.search_radius_approx, true)
                );
                for (const auto& i_d : indices_dist){
                    auto reg_it = pcd_to_reg_idxmap.find(i_d.first);
                    if (reg_it != pcd_to_reg_idxmap.end()){
                        candidate_regions.insert(reg_it->second);
                    }
                }
            }
            if (candidate_regions.empty()){
                continue;
            }

            float best_mean_dist = std::numeric_limits<float>::max();
            size_t best_region = 0;
            bool found = false;
            for (const auto& reg_id : candidate_regions){
                const Region& cand_region = regions[reg_id];
                float total_dist = 0.0f;
                for (const auto& idx : comp_inds){
                    total_dist += std::abs(
                        (pcd.coords[idx] - cand_region.centroid).dot(cand_region.normal)
                    );
                }
                const float mean_dist = total_dist / static_cast<float>(comp_inds.size());
                if (!found || mean_dist < best_mean_dist){
                    found = true;
                    best_mean_dist = mean_dist;
                    best_region = reg_id;
                }
            }

            if (found && best_mean_dist <= p.epsilon_multiplier * p.epsilon){
                for (const auto& idx : comp_inds){
                    regions[best_region].indices.insert(idx);
                    pcd_to_reg_idxmap[idx] = best_region;
                    unassigned_inds.erase(idx);
                }
                changed_regions.insert(best_region);
            }
        }

        for (const auto& reg_id : changed_regions){
            refitPlane(regions[reg_id], pcd);
        }
    }

    for (auto idx_it = unassigned_inds.begin(); idx_it != unassigned_inds.end();){
        IndicesDistType indices_dist;
        RadiusSetType result_set(p.epsilon, indices_dist);
        kdtree.findNeighbors(
            result_set,
            pcd.coords[*idx_it].data(),
            nanoflann::SearchParameters(p.search_radius_approx, true)
        );

        float min_dist_to_plane = std::numeric_limits<float>::max();
        float min_dist_criterion = std::numeric_limits<float>::max();
        size_t idx_best_region = 0;
        bool found = false;
        const float eps_safe = std::max(1e-8f, p.epsilon);
        const float eps_plane = std::max(1e-8f, p.epsilon * p.epsilon_multiplier);

        for (const auto& i_d : indices_dist){
            auto reg_it = pcd_to_reg_idxmap.find(i_d.first);
            if (reg_it == pcd_to_reg_idxmap.end()){
                continue;
            }
            const size_t reg_id = reg_it->second;
            const Region& cand_region = regions[reg_id];
            const float dist_to_plane = std::abs(
                (pcd.coords[*idx_it] - cand_region.centroid).dot(cand_region.normal)
            );
            const float dist_criterion =
                (static_cast<float>(i_d.second) / eps_safe) + (dist_to_plane / eps_plane);
            if (!found || dist_criterion < min_dist_criterion){
                found = true;
                min_dist_criterion = dist_criterion;
                min_dist_to_plane = dist_to_plane;
                idx_best_region = reg_id;
            }
        }

        if (found && min_dist_to_plane < p.epsilon * p.epsilon_multiplier){
            regions[idx_best_region].indices.insert(*idx_it);
            pcd_to_reg_idxmap[*idx_it] = idx_best_region; // point -> region
            idx_it = unassigned_inds.erase(idx_it);
        } else {
            ++idx_it;
        }
    }

    return regionGrowing_returnType{std::move(regions), pcd_to_reg_idxmap, unassigned_inds};
}
