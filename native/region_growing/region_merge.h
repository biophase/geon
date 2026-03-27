#pragma once

#include "types.h"
#include "nanoflann.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

using RegionMergeKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud>,
    PointCloud,
    3
>;

struct RegionMergeParams {
    float neighbor_radius = 0.05f;
    int32_t min_contact_points = 5;
    float planarity_threshold = 0.6f;
    float normal_angle_deg = 10.0f;
    float plane_distance_threshold = 0.03f;
    int32_t min_region_size = 20;
};

struct RegionGeometry {
    int32_t source_label = -1;
    size_t size = 0;
    Point centroid = Point::Zero();
    Point normal = Point{0.0f, 0.0f, 1.0f};
    Point eigenvalues = Point::Zero();
    float planarity = 0.0f;
    bool valid_plane = false;
};

struct RegionMergeStats {
    int32_t num_points = 0;
    int32_t num_input_regions = 0;
    int32_t num_planar_regions = 0;
    int32_t num_adjacency_pairs = 0;
    int32_t num_merge_candidates = 0;
    int32_t num_output_regions = 0;
    double elapsed_seconds = 0.0;
    bool cancelled = false;
};

struct RegionMergeProgressState {
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

struct RegionMergeResult {
    std::vector<int32_t> labels;
    RegionMergeStats stats;
};

RegionMergeResult merge_planar_regions_impl(
    const PointCloud& pcd,
    const std::vector<int32_t>& labels,
    const RegionMergeParams& params,
    RegionMergeProgressState* progress
);
