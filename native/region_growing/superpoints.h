#pragma once

#include "cp_d0_dist.hpp"
#include "nanoflann.hpp"
#include "types.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

using SuperpointKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud>,
    PointCloud,
    3
>;

struct SuperpointParams {
    int32_t k_neighbors = 10;
    float regularization = 0.05f;
    float spatial_weight = 1.0f;
    int32_t cutoff = 10;
    int32_t iterations = 10;
    bool parallel = true;
    bool verbose = false;
};

struct SuperpointStats {
    int32_t num_points = 0;
    int32_t num_superpoints = 0;
    int32_t feature_dim = 0;
    int32_t num_edges = 0;
    double elapsed_seconds = 0.0;
    bool cancelled = false;
};

struct SuperpointProgressState {
    std::atomic<int64_t> done{0};
    std::atomic<int64_t> total{0};
    std::atomic<bool> cancel{false};
    mutable std::mutex mutex;
    std::string stage = "Idle";

    void reset(int64_t total_count);
    void requestCancel();
    bool isCancelled() const;
    int64_t completed() const;
    int64_t totalCount() const;
    void setStage(const std::string& value, int64_t done_count);
    std::string stageText() const;
};

struct SuperpointResult {
    std::vector<int32_t> labels;
    SuperpointStats stats;
};

SuperpointResult segment_superpoints_impl(
    const PointCloud& pcd,
    const std::vector<float>& extra_features,
    int32_t extra_feature_dim,
    const SuperpointParams& params,
    SuperpointProgressState* progress
);
