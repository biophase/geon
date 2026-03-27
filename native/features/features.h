#pragma once
#include "Eigen/Dense"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
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


struct VoxelIndex{
    int64_t x;
    int64_t y;
    int64_t z;

    bool operator==(const VoxelIndex& other) const{
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelIndexHash{
    size_t operator()(const VoxelIndex& idx) const{
        const auto mix = [](uint64_t v) -> uint64_t {
            v += 0x9e3779b97f4a7c15ULL;
            v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
            v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
            return v ^ (v >> 31);
        };

        const uint64_t hx = mix(static_cast<uint64_t>(idx.x));
        const uint64_t hy = mix(static_cast<uint64_t>(idx.y));
        const uint64_t hz = mix(static_cast<uint64_t>(idx.z));
        return static_cast<size_t>(hx ^ (hy << 1) ^ (hz << 2));
    }
};

// hash map
using VoxelHash = std::unordered_map<VoxelIndex, std::vector<uint32_t>, VoxelIndexHash>;

struct Point{
    float x;
    float y;
    float z;
};

struct ProgressState{
    std::atomic<int64_t> done{0};
    std::atomic<int64_t> total{0};
    std::atomic<bool> cancel{false};

    void reset(int64_t total_count){
        done.store(0, std::memory_order_relaxed);
        total.store(total_count, std::memory_order_relaxed);
        cancel.store(false, std::memory_order_relaxed);
    }

    void requestCancel(){
        cancel.store(true, std::memory_order_relaxed);
    }

    bool isCancelled() const{
        return cancel.load(std::memory_order_relaxed);
    }

    int64_t completed() const{
        return done.load(std::memory_order_relaxed);
    }

    int64_t totalCount() const{
        return total.load(std::memory_order_relaxed);
    }
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
    MapMatNx3fRM& out_normals,
    ProgressState* progress
);
