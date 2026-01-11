#pragma once

#include "nanoflann.hpp"
#include "types.h"



/////////////////////
/* Nanoflann types */
/////////////////////

using KDTreeType = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud>,
    PointCloud,
    3 // 3D points
>;
using RadiusSetType = nanoflann::RadiusResultSet<float, size_t>;
using IndicesDistType = std::vector<nanoflann::ResultItem<size_t, float>>;


////////////////////////////
/* Region growing methods */
////////////////////////////
struct RegionGrowingParams;

void computeNormals(
    const std::vector<Point>& coords, 
    std::vector<Point>& normals,
    const std::unordered_set<size_t>& task_inds,
    KDTreeType& kdtree,
    float search_radius
    );


void processChunk(
    size_t chunk_id,
    const std::unordered_set<size_t>& chunk_indices,
    PointCloud& pcd,
    KDTreeType& kdtree,
    RegionGrowingParams params
);
void processChunkIndexContainer(
    size_t chunk_id,
    const std::unordered_set<size_t>& chunk_indices,
    PointCloud& pcd,
    KDTreeType& kdtree,
    RegionGrowingParams params,
    int32_t* output,
    std::unordered_map<size_t,size_t>& thread_offsets
);
struct regionGrowing_returnType{
    std::vector<Region> regions;
    std::unordered_map<size_t, size_t> pcd_to_reg_idxmap;
    std::unordered_set<size_t> unassigned;
};

regionGrowing_returnType regionGrowing( 
    PointCloud& pcd,
    KDTreeType& kdtree,
    const std::unordered_set<size_t>* task_indices, // indices to segment; if nullptr -> segment all
    RegionGrowingParams params
);

struct RegionGrowingParams{
    float epsilon = 0.03;
    float refit_multiplier = 2.f;
    float epsilon_multiplier = 3.f; // Value in Poux & Kobbelt paper = 3
    float epsilon_multiplier_average = 1.5f; // average residual should be bellow value * epsilon, otherwise stop growing
    float search_radius_approx = 0.f; // accuracy of radius search; 0. = exact sarch
    size_t min_points_in_region = 80;
    size_t first_refit = 4;
    float alpha = PI * 0.15;
    float min_success_ratio = 0.10f;
    float max_dist_from_cent = 50.f;
    size_t tracker_size = 50;
    bool oriented_normals = false;
    bool verbose = true;
    bool perform_cca = true; // divides each region in connected components after segmentation
};

/////////////////////
/* utility methods */
/////////////////////

template <typename T>
T randomPop (std::unordered_set<T>& input);

std::pair<Eigen::Vector3f, Eigen::Vector3f> getPcdAabb(const PointCloud& pcd);
std::pair<Eigen::Vector3f, Eigen::Vector3f> getPcdAabb(const PointCloud& pcd,
    std::unordered_set<size_t>& task_inds);

size_t flattenIndex (size_t x, size_t X, size_t y, size_t Y, size_t z, size_t Z);

std::vector<size_t> unflattenIndex(size_t flat_idx, size_t X, size_t Y, size_t Z);

std::unordered_map<size_t, std::unordered_set<size_t>> subdividePointCloud(const PointCloud& pcd, 
    size_t num_chunks_x, size_t num_chunks_y, size_t num_chunks_z);

struct subdividePointCloudFixedChunkSize_returnType{
    std::unordered_map<size_t, std::unordered_set<size_t>> chunks;
    size_t X, Y, Z;
};

subdividePointCloudFixedChunkSize_returnType subdividePointCloudFixedChunkSize(const PointCloud& pcd,
    std::unordered_set<size_t> task_inds, float chunk_size_x, float chunk_size_y, float chunk_size_z 
    );

class AttemptsTracker;