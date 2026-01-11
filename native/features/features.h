#pragma once
#include "Eigen/Dense"
#include <unordered_map>
#include <vector>

// numpy N,3
using MatNx3fRM = Eigen::Matrix<
    float, 
    Eigen::Dynamic, 
    3, 
    Eigen::RowMajor>;
using MapMatNx3fRM = Eigen::Map<MatNx3fRM>;

// numpy N,M
using MatNxMfRM = Eigen::Matrix<
    float, 
    Eigen::Dynamic, 
    Eigen::Dynamic, 
    Eigen::RowMajor>;
using MapMatNxMfRM = Eigen::Map<MatNxMfRM>;

// numpy N,
using VecNuint64RM = Eigen::Matrix<
    uint64_t,
    Eigen::Dynamic, 
    1>;
using MapVecNuint64RM = Eigen::Map<VecNuint64RM>;


// hash map
using VoxelHash = std::unordered_map<uint64_t, std::vector<uint32_t>>;

struct Point{
    float x;
    float y;
    float z;
};

uint64_t voxelKey(float x, float y, float z, float inv_s);

VoxelHash computeVoxelHash(
    MapMatNx3fRM& positive_coords,
    float& inv_s
);

std::vector<uint32_t> getNeighborIndsRadius(
    float radius,
    Point query,
    float voxel_size,
    const VoxelHash& voxel_hash,
    const MapMatNx3fRM& positive_coords
);

void computePcdFeatures(
    float radius,
    float voxel_size,
    const MapMatNx3fRM& positive_coords,
    const VoxelHash& voxel_hash,
    MapMatNx3fRM& out_eigenvalues,
    MapMatNx3fRM& out_normals
);
