#include "superpoints.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace {

#if defined _OPENMP && _OPENMP < 200805
using index_t = int32_t;
using comp_t = int32_t;
#else
using index_t = uint32_t;
using comp_t = uint32_t;
#endif

struct DirectedEdge {
    index_t src;
    index_t dst;

    bool operator<(const DirectedEdge& other) const{
        if (src != other.src){
            return src < other.src;
        }
        return dst < other.dst;
    }

    bool operator==(const DirectedEdge& other) const{
        return src == other.src && dst == other.dst;
    }
};

void build_knn_edges(
    const PointCloud& pcd,
    SuperpointKDTree& kdtree,
    int32_t k_neighbors,
    std::vector<index_t>& first_edge,
    std::vector<index_t>& adj_vertices,
    int32_t& num_edges_out
){
    const size_t n_points = pcd.coords.size();
    const int32_t k = std::max<int32_t>(1, std::min<int32_t>(k_neighbors, static_cast<int32_t>(n_points > 0 ? n_points - 1 : 0)));
    if (n_points == 0){
        first_edge = {0};
        adj_vertices.clear();
        num_edges_out = 0;
        return;
    }
    if (n_points == 1 || k == 0){
        first_edge.assign(n_points + 1, 0);
        adj_vertices.clear();
        num_edges_out = 0;
        return;
    }

    std::vector<DirectedEdge> edges;
    edges.reserve(static_cast<size_t>(n_points) * static_cast<size_t>(k) * 2);

    std::vector<uint32_t> ret_indexes(static_cast<size_t>(k + 1));
    std::vector<float> out_dists(static_cast<size_t>(k + 1));

    for (size_t i = 0; i < n_points; ++i){
        const Point& p = pcd.coords[i];
        const size_t found = kdtree.knnSearch(
            p.data(),
            static_cast<size_t>(k + 1),
            ret_indexes.data(),
            out_dists.data()
        );
        for (size_t j = 0; j < found; ++j){
            const size_t neighbor = ret_indexes[j];
            if (neighbor == i){
                continue;
            }
            const auto src = static_cast<index_t>(i);
            const auto dst = static_cast<index_t>(neighbor);
            edges.push_back({src, dst});
            edges.push_back({dst, src});
        }
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    first_edge.assign(n_points + 1, 0);
    adj_vertices.resize(edges.size());

    size_t edge_pos = 0;
    for (size_t src = 0; src < n_points; ++src){
        first_edge[src] = static_cast<index_t>(edge_pos);
        while (edge_pos < edges.size() && edges[edge_pos].src == src){
            adj_vertices[edge_pos] = edges[edge_pos].dst;
            ++edge_pos;
        }
    }
    first_edge[n_points] = static_cast<index_t>(edges.size());
    num_edges_out = static_cast<int32_t>(edges.size());
}

std::vector<float> build_feature_matrix(
    const PointCloud& pcd,
    const std::vector<float>& extra_features,
    int32_t extra_feature_dim,
    float spatial_weight,
    int32_t& total_dim
){
    const size_t n_points = pcd.coords.size();
    if (extra_feature_dim < 0){
        throw std::runtime_error("extra_feature_dim must be non-negative");
    }
    if (extra_feature_dim > 0 &&
        extra_features.size() != n_points * static_cast<size_t>(extra_feature_dim)){
        throw std::runtime_error("extra_features shape does not match coords row count");
    }

    total_dim = 3 + extra_feature_dim;
    std::vector<float> y(static_cast<size_t>(total_dim) * n_points, 0.0f);

    Point mean = Point::Zero();
    if (!pcd.coords.empty()){
        for (const Point& p : pcd.coords){
            mean += p;
        }
        mean /= static_cast<float>(pcd.coords.size());
    }

    for (size_t i = 0; i < n_points; ++i){
        const Point centered = pcd.coords[i] - mean;
        y[static_cast<size_t>(0) + static_cast<size_t>(total_dim) * i] = centered.x();
        y[static_cast<size_t>(1) + static_cast<size_t>(total_dim) * i] = centered.y();
        y[static_cast<size_t>(2) + static_cast<size_t>(total_dim) * i] = centered.z();
        for (int32_t d = 0; d < extra_feature_dim; ++d){
            y[static_cast<size_t>(3 + d) + static_cast<size_t>(total_dim) * i] =
                extra_features[i * static_cast<size_t>(extra_feature_dim) + static_cast<size_t>(d)];
        }
    }

    (void) spatial_weight;
    return y;
}

} // namespace

void SuperpointProgressState::reset(int64_t total_count){
    done.store(0, std::memory_order_relaxed);
    total.store(total_count, std::memory_order_relaxed);
    cancel.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex);
    stage = "Idle";
}

void SuperpointProgressState::requestCancel(){
    cancel.store(true, std::memory_order_relaxed);
}

bool SuperpointProgressState::isCancelled() const{
    return cancel.load(std::memory_order_relaxed);
}

int64_t SuperpointProgressState::completed() const{
    return done.load(std::memory_order_relaxed);
}

int64_t SuperpointProgressState::totalCount() const{
    return total.load(std::memory_order_relaxed);
}

void SuperpointProgressState::setStage(const std::string& value, int64_t done_count){
    done.store(done_count, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex);
    stage = value;
}

std::string SuperpointProgressState::stageText() const{
    std::lock_guard<std::mutex> lock(mutex);
    return stage;
}

SuperpointResult segment_superpoints_impl(
    const PointCloud& pcd,
    const std::vector<float>& extra_features,
    int32_t extra_feature_dim,
    const SuperpointParams& params,
    SuperpointProgressState* progress
){
    const auto t0 = std::chrono::steady_clock::now();
    const size_t n_points = pcd.coords.size();

    SuperpointResult result;
    result.labels.assign(n_points, -1);
    result.stats.num_points = static_cast<int32_t>(n_points);

    if (progress != nullptr){
        progress->reset(3);
        progress->setStage("Building graph...", 0);
    }
    if (progress != nullptr && progress->isCancelled()){
        result.stats.cancelled = true;
        return result;
    }

    if (n_points == 0){
        result.stats.feature_dim = 0;
        if (progress != nullptr){
            progress->setStage("Done", 3);
        }
        return result;
    }

    PointCloud tree_cloud;
    tree_cloud.coords = pcd.coords;
    SuperpointKDTree kdtree(3, tree_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();

    std::vector<index_t> first_edge;
    std::vector<index_t> adj_vertices;
    build_knn_edges(
        tree_cloud,
        kdtree,
        params.k_neighbors,
        first_edge,
        adj_vertices,
        result.stats.num_edges
    );

    int32_t total_dim = 0;
    std::vector<float> y = build_feature_matrix(
        pcd,
        extra_features,
        extra_feature_dim,
        params.spatial_weight,
        total_dim
    );
    result.stats.feature_dim = total_dim;

    std::vector<float> coor_weights(static_cast<size_t>(total_dim), 1.0f);
    coor_weights[0] = params.spatial_weight;
    coor_weights[1] = params.spatial_weight;
    coor_weights[2] = params.spatial_weight;
    std::vector<float> vert_weights(n_points, 1.0f);

    if (progress != nullptr){
        progress->setStage("Running cut pursuit...", 1);
    }
    if (progress != nullptr && progress->isCancelled()){
        result.stats.cancelled = true;
        return result;
    }

    std::vector<comp_t> comp_assign(n_points, 0);
    {
        Cp_d0_dist<float, index_t, comp_t> cp(
            static_cast<index_t>(n_points),
            static_cast<index_t>(adj_vertices.size()),
            first_edge.data(),
            adj_vertices.data(),
            y.data(),
            static_cast<size_t>(total_dim)
        );
        cp.set_loss(
            static_cast<float>(total_dim),
            y.data(),
            vert_weights.data(),
            coor_weights.data()
        );
        cp.set_edge_weights(nullptr, params.regularization);
        cp.set_cp_param(1e-2f, std::max(1, params.iterations), params.verbose ? 1 : 0);
        cp.set_split_param(
            static_cast<index_t>(n_points),
            static_cast<comp_t>(2),
            2,
            0.7f,
            3,
            3
        );
        cp.set_min_comp_weight(static_cast<float>(std::max(1, params.cutoff)));
        cp.set_parallel_param(params.parallel ? omp_get_max_threads() : 1, true);
        cp.set_components(0, comp_assign.data());
        cp.cut_pursuit();
        cp.set_components(0, nullptr);
    }

    if (progress != nullptr){
        progress->setStage("Writing labels...", 2);
    }

    int32_t max_label = -1;
    for (size_t i = 0; i < n_points; ++i){
        result.labels[i] = static_cast<int32_t>(comp_assign[i]);
        if (result.labels[i] > max_label){
            max_label = result.labels[i];
        }
    }
    result.stats.num_superpoints = max_label + 1;

    const auto t1 = std::chrono::steady_clock::now();
    result.stats.elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    if (progress != nullptr){
        progress->setStage("Done", 3);
    }
    return result;
}
