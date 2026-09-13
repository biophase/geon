#pragma once

#include "nanoflann.hpp"
#include "types.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

using SplitBetweenKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3>;

enum class SplitBetweenCondition { PlaneDistance, NearestNeighbor };

struct SplitBetweenProgressState {
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

struct SplitBetweenResult {
    std::vector<uint8_t> assignment;
    int64_t assigned_to_a = 0;
    int64_t assigned_to_b = 0;
    double elapsed_seconds = 0.0;
    bool cancelled = false;
};

SplitBetweenResult split_between_impl(
    const PointCloud& pcd,
    const std::vector<size_t>& working_indices,
    const std::vector<size_t>& a_indices,
    const std::vector<size_t>& b_indices,
    SplitBetweenCondition condition,
    SplitBetweenProgressState* progress
);
