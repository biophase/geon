#pragma once

#include "nanoflann.hpp"
#include "types.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

using CornerCleanupKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud>,
    PointCloud,
    3
>;

struct CornerCleanupParams {
    float epsilon = 0.03f;
    float neighbor_radius_factor = 3.0f;
    float slenderness_threshold = 0.02f;
    float absolute_width_factor = 3.0f;
    float relative_width_factor = 0.5f;
    float dual_support_threshold = 0.5f;
    int32_t min_corner_size = 10;
    int32_t min_planar_size = 20;
};

struct CornerCleanupStats {
    int32_t num_points = 0;
    int32_t num_input_regions = 0;
    int32_t num_destination_regions = 0;
    float neighbor_radius = 0.0f;
    int32_t num_corner_candidates = 0;
    int32_t num_rejected_slenderness = 0;
    int32_t num_rejected_absolute_width = 0;
    int32_t num_rejected_relative_width = 0;
    int32_t num_rejected_planar_neighbors = 0;
    int32_t num_rejected_dual_support = 0;
    int32_t num_corner_regions_cleaned = 0;
    int32_t num_corner_points_reassigned = 0;
    double elapsed_seconds = 0.0;
    bool cancelled = false;
};

struct CornerCleanupProgressState {
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

struct CornerCleanupResult {
    std::vector<int32_t> labels;
    CornerCleanupStats stats;
};

CornerCleanupResult cleanup_corner_regions_impl(
    const PointCloud& pcd,
    const std::vector<int32_t>& labels,
    const CornerCleanupParams& params,
    CornerCleanupProgressState* progress
);
