#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "Eigen/Dense"

using SubsampleMatNx3fRM = Eigen::Matrix<
    float,
    Eigen::Dynamic,
    3,
    Eigen::RowMajor>;
using SubsampleMapMatNx3fRM = Eigen::Map<SubsampleMatNx3fRM>;
using SubsampleRefMatNx3fRM = Eigen::Ref<const SubsampleMatNx3fRM>;

struct SubsampleVoxelIndex{
    int64_t x;
    int64_t y;
    int64_t z;

    bool operator==(const SubsampleVoxelIndex& other) const{
        return x == other.x && y == other.y && z == other.z;
    }
};

struct SubsampleVoxelIndexHash{
    size_t operator()(const SubsampleVoxelIndex& idx) const;
};

using SubsampleVoxelHash = std::unordered_map<
    SubsampleVoxelIndex,
    std::vector<uint32_t>,
    SubsampleVoxelIndexHash>;

SubsampleVoxelHash computeSubsampleVoxelHash(
    SubsampleRefMatNx3fRM positive_coords,
    float voxel_size
);

std::vector<uint8_t> spatialSubsampleMask(
    SubsampleRefMatNx3fRM coords,
    float min_distance
);
