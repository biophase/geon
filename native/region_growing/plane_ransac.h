#pragma once

#include "types.h"
#include "nanoflann.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using PlaneRansacKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud>,
    PointCloud,
    3
>;
using PlaneRansacRadiusSet = nanoflann::RadiusResultSet<float, size_t>;
using PlaneRansacIndicesDist = std::vector<nanoflann::ResultItem<size_t, float>>;

struct PlaneRansacParams {
    float epsilon = 0.03f;
    size_t min_points = 100;
    float normal_threshold_deg = 25.0f;
    float cluster_epsilon = -1.0f;
    float probability = 0.01f;
    size_t max_iterations_per_plane = 5000;
    uint32_t seed = 0;
};

struct PlaneModel {
    Point centroid = Point{0.0f, 0.0f, 0.0f};
    Point normal = Point{0.0f, 0.0f, 1.0f};
    float d = 0.0f;
    std::vector<size_t> indices;
};

struct PlaneRansacRuntime {
    std::string stage;
    size_t planes_found = 0;
    size_t active_points_remaining = 0;
    size_t current_best_support = 0;
    size_t assigned_points = 0;
    bool done = false;
};

struct PlaneRansacResult {
    std::vector<int32_t> labels;
    std::vector<PlaneModel> planes;
    bool cancelled = false;
};

using PlaneRansacProgressCallback = std::function<bool(const PlaneRansacRuntime&)>;

void compute_plane_ransac_normals(
    const std::vector<Point>& coords,
    std::vector<Point>& normals,
    PlaneRansacKDTree& kdtree,
    float search_radius
);

PlaneRansacResult segment_planes_ransac(
    PointCloud& pcd,
    PlaneRansacKDTree& kdtree,
    const PlaneRansacParams& params,
    const PlaneRansacProgressCallback& progress_cb = {}
);
