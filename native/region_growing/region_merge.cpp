#include "region_merge.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "Eigen/Dense"

namespace {

struct RegionPairHash {
    size_t operator()(const std::pair<int32_t, int32_t>& p) const{
        return (static_cast<uint64_t>(static_cast<uint32_t>(p.first)) << 32U)
             ^ static_cast<uint32_t>(p.second);
    }
};

struct UnionFind {
    std::vector<int32_t> parent;
    std::vector<int32_t> rank;

    explicit UnionFind(size_t n)
        : parent(n), rank(n, 0)
    {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int32_t find(int32_t x){
        if (parent[static_cast<size_t>(x)] != x){
            parent[static_cast<size_t>(x)] = find(parent[static_cast<size_t>(x)]);
        }
        return parent[static_cast<size_t>(x)];
    }

    void unite(int32_t a, int32_t b){
        a = find(a);
        b = find(b);
        if (a == b){
            return;
        }
        if (rank[static_cast<size_t>(a)] < rank[static_cast<size_t>(b)]){
            std::swap(a, b);
        }
        parent[static_cast<size_t>(b)] = a;
        if (rank[static_cast<size_t>(a)] == rank[static_cast<size_t>(b)]){
            ++rank[static_cast<size_t>(a)];
        }
    }
};

float clamp_dot(float value){
    return std::max(-1.0f, std::min(1.0f, value));
}

float compute_planarity(const Point& eigenvalues){
    const float l1 = eigenvalues[0];
    const float l2 = eigenvalues[1];
    const float l3 = eigenvalues[2];
    const float sum = l1 + l2 + l3;
    if (sum <= 1e-8f){
        return 0.0f;
    }
    // Region-level "planarity" here is used as a flatness score:
    // planar patches should have very small normal-direction variance,
    // even if they are anisotropic within the plane.
    return std::max(0.0f, 1.0f - (l1 / sum));
}

std::vector<RegionGeometry> compute_region_geometry(
    const PointCloud& pcd,
    const std::vector<int32_t>& labels,
    const std::vector<int32_t>& unique_labels,
    const std::unordered_map<int32_t, int32_t>& label_to_region
){
    std::vector<std::vector<size_t>> members(unique_labels.size());
    for (size_t i = 0; i < labels.size(); ++i){
        const int32_t label = labels[i];
        if (label < 0){
            continue;
        }
        const auto it = label_to_region.find(label);
        if (it != label_to_region.end()){
            members[static_cast<size_t>(it->second)].push_back(i);
        }
    }

    std::vector<RegionGeometry> regions(unique_labels.size());
    for (size_t region_idx = 0; region_idx < unique_labels.size(); ++region_idx){
        RegionGeometry& geom = regions[region_idx];
        geom.source_label = unique_labels[region_idx];
        const auto& inds = members[region_idx];
        geom.size = inds.size();
        if (inds.empty()){
            continue;
        }

        Point centroid = Point::Zero();
        for (const size_t idx : inds){
            centroid += pcd.coords[idx];
        }
        centroid /= static_cast<float>(inds.size());
        geom.centroid = centroid;

        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        for (const size_t idx : inds){
            const Point d = pcd.coords[idx] - centroid;
            cov += d * d.transpose();
        }
        cov /= static_cast<float>(inds.size());

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
        if (solver.info() != Eigen::Success){
            continue;
        }

        geom.eigenvalues = solver.eigenvalues();
        geom.normal = solver.eigenvectors().col(0).normalized();
        geom.planarity = compute_planarity(geom.eigenvalues);
        geom.valid_plane = std::isfinite(geom.planarity)
            && geom.normal.allFinite()
            && geom.size > 0;
    }

    return regions;
}

bool compatible_for_merge(
    const RegionGeometry& a,
    const RegionGeometry& b,
    const RegionMergeParams& params
){
    if (!a.valid_plane || !b.valid_plane){
        return false;
    }
    if (static_cast<int32_t>(a.size) < params.min_region_size ||
        static_cast<int32_t>(b.size) < params.min_region_size){
        return false;
    }
    if (a.planarity < params.planarity_threshold ||
        b.planarity < params.planarity_threshold){
        return false;
    }

    const float cos_angle = std::abs(clamp_dot(a.normal.dot(b.normal)));
    const float angle_deg = std::acos(cos_angle) * 180.0f / static_cast<float>(PI);
    if (angle_deg > params.normal_angle_deg){
        return false;
    }

    const float dist_ab = std::abs(a.normal.dot(b.centroid - a.centroid));
    const float dist_ba = std::abs(b.normal.dot(a.centroid - b.centroid));
    const float sym_dist = std::max(dist_ab, dist_ba);
    return sym_dist <= params.plane_distance_threshold;
}

} // namespace

void RegionMergeProgressState::reset(int64_t total_count){
    done.store(0, std::memory_order_relaxed);
    total.store(total_count, std::memory_order_relaxed);
    cancel.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex);
    stage = "Idle";
}

void RegionMergeProgressState::requestCancel(){
    cancel.store(true, std::memory_order_relaxed);
}

bool RegionMergeProgressState::isCancelled() const{
    return cancel.load(std::memory_order_relaxed);
}

int64_t RegionMergeProgressState::completed() const{
    return done.load(std::memory_order_relaxed);
}

int64_t RegionMergeProgressState::totalCount() const{
    return total.load(std::memory_order_relaxed);
}

void RegionMergeProgressState::setStage(const std::string& value, int64_t done_count){
    done.store(done_count, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex);
    stage = value;
}

std::string RegionMergeProgressState::stageText() const{
    std::lock_guard<std::mutex> lock(mutex);
    return stage;
}

RegionMergeResult merge_planar_regions_impl(
    const PointCloud& pcd,
    const std::vector<int32_t>& labels,
    const RegionMergeParams& params,
    RegionMergeProgressState* progress
){
    const auto t0 = std::chrono::steady_clock::now();
    const size_t n_points = pcd.coords.size();
    if (labels.size() != n_points){
        throw std::runtime_error("labels length does not match point count");
    }

    RegionMergeResult result;
    result.labels.assign(labels.begin(), labels.end());
    result.stats.num_points = static_cast<int32_t>(n_points);

    if (progress != nullptr){
        progress->reset(4);
        progress->setStage("Computing region geometry...", 0);
    }

    std::vector<int32_t> unique_labels;
    unique_labels.reserve(labels.size());
    {
        std::unordered_map<int32_t, bool> seen;
        for (const int32_t label : labels){
            if (label < 0){
                continue;
            }
            if (seen.find(label) == seen.end()){
                seen[label] = true;
                unique_labels.push_back(label);
            }
        }
    }
    std::sort(unique_labels.begin(), unique_labels.end());
    result.stats.num_input_regions = static_cast<int32_t>(unique_labels.size());

    std::unordered_map<int32_t, int32_t> label_to_region;
    label_to_region.reserve(unique_labels.size());
    for (size_t i = 0; i < unique_labels.size(); ++i){
        label_to_region[unique_labels[i]] = static_cast<int32_t>(i);
    }

    std::vector<RegionGeometry> regions = compute_region_geometry(
        pcd,
        labels,
        unique_labels,
        label_to_region
    );
    for (const RegionGeometry& region : regions){
        if (region.valid_plane &&
            static_cast<int32_t>(region.size) >= params.min_region_size &&
            region.planarity >= params.planarity_threshold){
            ++result.stats.num_planar_regions;
        }
    }

    if (progress != nullptr){
        if (progress->isCancelled()){
            result.stats.cancelled = true;
            return result;
        }
        progress->setStage("Building adjacency graph...", 1);
    }

    PointCloud tree_cloud;
    tree_cloud.coords = pcd.coords;
    RegionMergeKDTree kdtree(3, tree_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();

    std::unordered_map<std::pair<int32_t, int32_t>, int32_t, RegionPairHash> contacts;
    const float radius_sq = params.neighbor_radius * params.neighbor_radius;
    for (size_t i = 0; i < n_points; ++i){
        if (progress != nullptr && (i % 10000 == 0) && progress->isCancelled()){
            result.stats.cancelled = true;
            return result;
        }
        const int32_t label_i = labels[i];
        if (label_i < 0){
            continue;
        }
        std::vector<nanoflann::ResultItem<size_t, float>> neighbors;
        nanoflann::RadiusResultSet<float, size_t> result_set(radius_sq, neighbors);
        kdtree.findNeighbors(result_set, pcd.coords[i].data(), nanoflann::SearchParameters());
        for (const auto& item : neighbors){
            const size_t j = item.first;
            if (j == i){
                continue;
            }
            const int32_t label_j = labels[j];
            if (label_j < 0 || label_j == label_i){
                continue;
            }
            int32_t a = label_i;
            int32_t b = label_j;
            if (a > b){
                std::swap(a, b);
            }
            ++contacts[{a, b}];
        }
    }
    result.stats.num_adjacency_pairs = static_cast<int32_t>(contacts.size());

    if (progress != nullptr){
        if (progress->isCancelled()){
            result.stats.cancelled = true;
            return result;
        }
        progress->setStage("Merging compatible planar regions...", 2);
    }

    UnionFind uf(unique_labels.size());
    for (const auto& [pair, count] : contacts){
        if (count < params.min_contact_points){
            continue;
        }
        const RegionGeometry& a = regions[static_cast<size_t>(label_to_region.at(pair.first))];
        const RegionGeometry& b = regions[static_cast<size_t>(label_to_region.at(pair.second))];
        if (!compatible_for_merge(a, b, params)){
            continue;
        }
        uf.unite(label_to_region.at(pair.first), label_to_region.at(pair.second));
        ++result.stats.num_merge_candidates;
    }

    if (progress != nullptr){
        if (progress->isCancelled()){
            result.stats.cancelled = true;
            return result;
        }
        progress->setStage("Writing merged labels...", 3);
    }

    std::unordered_map<int32_t, int32_t> root_to_compact;
    int32_t next_label = 0;
    for (size_t i = 0; i < result.labels.size(); ++i){
        const int32_t source_label = labels[i];
        if (source_label < 0){
            result.labels[i] = -1;
            continue;
        }
        const int32_t region_idx = label_to_region.at(source_label);
        const int32_t root = uf.find(region_idx);
        auto [it, inserted] = root_to_compact.emplace(root, next_label);
        if (inserted){
            ++next_label;
        }
        result.labels[i] = it->second;
    }
    result.stats.num_output_regions = next_label;

    const auto t1 = std::chrono::steady_clock::now();
    result.stats.elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    if (progress != nullptr){
        progress->setStage("Done", 4);
    }
    return result;
}
