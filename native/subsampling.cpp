#include "subsampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

inline int64_t voxel_index(float x, float inv_s){
    return static_cast<int64_t>(std::floor(static_cast<double>(x) * static_cast<double>(inv_s)));
}

inline SubsampleVoxelIndex voxelIndex(float x, float y, float z, float inv_s){
    return SubsampleVoxelIndex{
        voxel_index(x, inv_s),
        voxel_index(y, inv_s),
        voxel_index(z, inv_s),
    };
}

} // namespace

size_t SubsampleVoxelIndexHash::operator()(const SubsampleVoxelIndex& idx) const{
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

SubsampleVoxelHash computeSubsampleVoxelHash(
    SubsampleRefMatNx3fRM positive_coords,
    float voxel_size
){
    if (!(voxel_size > 0.0f)){
        throw std::runtime_error("voxel_size must be > 0");
    }

    const float inv_s = 1.0f / voxel_size;
    SubsampleVoxelHash map;
    const auto n_rows = positive_coords.rows();
    map.reserve(static_cast<size_t>(n_rows));
    for (uint32_t i = 0; i < static_cast<uint32_t>(n_rows); ++i){
        const SubsampleVoxelIndex key = voxelIndex(
            positive_coords(i, 0),
            positive_coords(i, 1),
            positive_coords(i, 2),
            inv_s
        );
        map[key].push_back(i);
    }
    return map;
}

std::vector<uint8_t> spatialSubsampleMask(
    SubsampleRefMatNx3fRM coords,
    float min_distance
){
    if (!(min_distance > 0.0f)){
        throw std::runtime_error("min_distance must be > 0");
    }

    const auto n_rows = coords.rows();
    std::vector<uint8_t> survivors(static_cast<size_t>(n_rows), uint8_t{0});
    if (n_rows == 0){
        return survivors;
    }

    Eigen::RowVector3f min_corner = coords.colwise().minCoeff();
    SubsampleMatNx3fRM positive_coords = coords.rowwise() - min_corner;
    const float voxel_size = min_distance;
    const float inv_s = 1.0f / voxel_size;
    const float radius_sq = min_distance * min_distance;

    const SubsampleVoxelHash voxel_hash = computeSubsampleVoxelHash(positive_coords, voxel_size);
    std::vector<uint8_t> alive(static_cast<size_t>(n_rows), uint8_t{1});

    for (int64_t i = 0; i < n_rows; ++i){
        if (alive[static_cast<size_t>(i)] == 0){
            continue;
        }

        survivors[static_cast<size_t>(i)] = 1;
        const float qx = positive_coords(i, 0);
        const float qy = positive_coords(i, 1);
        const float qz = positive_coords(i, 2);

        const int64_t ix_min = voxel_index(qx - min_distance, inv_s);
        const int64_t iy_min = voxel_index(qy - min_distance, inv_s);
        const int64_t iz_min = voxel_index(qz - min_distance, inv_s);
        const int64_t ix_max = voxel_index(qx + min_distance, inv_s);
        const int64_t iy_max = voxel_index(qy + min_distance, inv_s);
        const int64_t iz_max = voxel_index(qz + min_distance, inv_s);

        for (int64_t ix = ix_min; ix <= ix_max; ++ix){
            for (int64_t iy = iy_min; iy <= iy_max; ++iy){
                for (int64_t iz = iz_min; iz <= iz_max; ++iz){
                    const SubsampleVoxelIndex key{ix, iy, iz};
                    const auto it = voxel_hash.find(key);
                    if (it == voxel_hash.end()){
                        continue;
                    }

                    for (uint32_t idx : it->second){
                        if (idx == static_cast<uint32_t>(i)){
                            continue;
                        }
                        if (alive[idx] == 0){
                            continue;
                        }

                        const float dx = positive_coords(static_cast<Eigen::Index>(idx), 0) - qx;
                        const float dy = positive_coords(static_cast<Eigen::Index>(idx), 1) - qy;
                        const float dz = positive_coords(static_cast<Eigen::Index>(idx), 2) - qz;
                        const float dist_sq = dx * dx + dy * dy + dz * dz;
                        if (dist_sq <= radius_sq){
                            alive[idx] = 0;
                        }
                    }
                }
            }
        }
    }

    return survivors;
}
