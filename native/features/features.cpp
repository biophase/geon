#include "features.h"

namespace {

inline int64_t voxel_index(float x, float inv_s){
    return static_cast<int64_t>(std::floor(static_cast<double>(x) * static_cast<double>(inv_s)));
}

inline VoxelIndex voxelIndex(float x, float y, float z, float inv_s){
    return VoxelIndex{
        voxel_index(x, inv_s),
        voxel_index(y, inv_s),
        voxel_index(z, inv_s),
    };
}

} // namespace

uint64_t voxelKey(float x, float y, float z, float inv_s){
        const VoxelIndex idx = voxelIndex(x, y, z, inv_s);
        VoxelIndexHash hasher;
        return static_cast<uint64_t>(hasher(idx));
}

VoxelHash computeVoxelHash(
    MapMatNx3fRM& positive_coords,
    float& inv_s

){
    
    VoxelHash map;
    const auto N = positive_coords.rows();
    for (uint32_t i = 0; i < N; ++i){
        VoxelIndex key = voxelIndex(
            positive_coords(i,0),
            positive_coords(i,1),
            positive_coords(i,2),
            inv_s
        );
        map[key].push_back(i);
    }
    return map;
}

std::vector<uint32_t> getNeighborIndsRadius(
    float radius,
    Point query,
    float voxel_size,
    const VoxelHash& voxel_hash,
    const MapMatNx3fRM& positive_coords
){
    std::vector<uint32_t> neighbors;
    if (radius <= 0.0f || voxel_size <= 0.0f || voxel_hash.empty()){
        return neighbors;
    }

    const float inv_s = 1.0f / voxel_size;
    const float r2 = radius * radius;

    const float min_x = query.x - radius;
    const float min_y = query.y - radius;
    const float min_z = query.z - radius;
    const float max_x = query.x + radius;
    const float max_y = query.y + radius;
    const float max_z = query.z + radius;

    const int64_t ix_min = voxel_index(min_x, inv_s);
    const int64_t iy_min = voxel_index(min_y, inv_s);
    const int64_t iz_min = voxel_index(min_z, inv_s);
    const int64_t ix_max = voxel_index(max_x, inv_s);
    const int64_t iy_max = voxel_index(max_y, inv_s);
    const int64_t iz_max = voxel_index(max_z, inv_s);

    for (int64_t ix = ix_min; ix <= ix_max; ++ix){
        for (int64_t iy = iy_min; iy <= iy_max; ++iy){
            for (int64_t iz = iz_min; iz <= iz_max; ++iz){
                const VoxelIndex key{ix, iy, iz};
                auto it = voxel_hash.find(key);
                if (it == voxel_hash.end()){
                    continue;
                }

                const auto& inds = it->second;
                for (uint32_t idx : inds){
                    const float dx = positive_coords(idx, 0) - query.x;
                    const float dy = positive_coords(idx, 1) - query.y;
                    const float dz = positive_coords(idx, 2) - query.z;
                    const float dist2 = dx * dx + dy * dy + dz * dz;
                    if (dist2 <= r2){
                        neighbors.push_back(idx);
                    }
                }
            }
        }
    }

    return neighbors;
}

void computePcdFeatures(
    float radius,
    float voxel_size,
    const MapMatNx3fRM& positive_coords,
    const VoxelHash& voxel_hash,
    MapMatNx3fRM& out_eigenvalues,
    MapMatNx3fRM& out_normals,
    ProgressState* progress
){
    const auto N = positive_coords.rows();
    if (progress != nullptr){
        progress->reset(static_cast<int64_t>(N));
    }

#pragma omp parallel
    {
        int64_t local_done = 0;

#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i){
            if (progress != nullptr && progress->isCancelled()){
                continue;
            }
        Point query{
            positive_coords(i, 0),
            positive_coords(i, 1),
            positive_coords(i, 2)
        };
        std::vector<uint32_t> neighbors = getNeighborIndsRadius(
            radius,
            query,
            voxel_size,
            voxel_hash,
            positive_coords
        );

        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (uint32_t idx : neighbors){
            const float dx = positive_coords(idx, 0) - query.x;
            const float dy = positive_coords(idx, 1) - query.y;
            const float dz = positive_coords(idx, 2) - query.z;
            Eigen::Vector3f delta(dx, dy, dz);
            covariance += delta * delta.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        out_eigenvalues.row(i) = solver.eigenvalues().transpose();
        out_normals.row(i) = solver.eigenvectors().col(0).transpose();

            if (progress != nullptr){
                local_done += 1;
                if ((local_done & 0x3f) == 0){
                    progress->done.fetch_add(local_done, std::memory_order_relaxed);
                    local_done = 0;
                }
            }
        }

        if (progress != nullptr && local_done > 0){
            progress->done.fetch_add(local_done, std::memory_order_relaxed);
        }
    }
}
