#include "connected_components.h"
#include "rgrow.h"
#include "types.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

struct ProgressState {
    std::atomic<int64_t> done{0};
    std::atomic<int64_t> total{0};
    std::atomic<bool> cancel{false};
    mutable std::mutex chunk_mutex;
    std::vector<int64_t> chunk_attempts;
    std::vector<int64_t> chunk_regions;
    std::vector<int64_t> chunk_remaining;
    std::vector<double> chunk_fail_rate;
    std::vector<double> chunk_fail_threshold;
    std::vector<uint8_t> chunk_phase;

    void reset(int64_t total_count){
        done.store(0, std::memory_order_relaxed);
        total.store(total_count, std::memory_order_relaxed);
        cancel.store(false, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(chunk_mutex);
        chunk_attempts.clear();
        chunk_regions.clear();
        chunk_remaining.clear();
        chunk_fail_rate.clear();
        chunk_fail_threshold.clear();
        chunk_phase.clear();
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

    void initChunks(size_t n_chunks){
        std::lock_guard<std::mutex> lock(chunk_mutex);
        chunk_attempts.assign(n_chunks, 0);
        chunk_regions.assign(n_chunks, 0);
        chunk_remaining.assign(n_chunks, 0);
        chunk_fail_rate.assign(n_chunks, 0.0);
        chunk_fail_threshold.assign(n_chunks, 0.9);
        chunk_phase.assign(n_chunks, 0);
    }

    void updateChunk(
        size_t idx,
        int64_t attempts,
        int64_t regions,
        int64_t remaining,
        double fail_rate,
        double fail_threshold,
        bool finished
    ){
        std::lock_guard<std::mutex> lock(chunk_mutex);
        if (idx >= chunk_attempts.size()){
            return;
        }
        chunk_attempts[idx] = attempts;
        chunk_regions[idx] = regions;
        chunk_remaining[idx] = remaining;
        chunk_fail_rate[idx] = fail_rate;
        chunk_fail_threshold[idx] = fail_threshold;
        if (finished){
            chunk_phase[idx] = 1; // finalizing
        } else if (chunk_phase[idx] == 0) {
            chunk_phase[idx] = 0; // segmenting
        }
    }

    void completeChunk(size_t idx){
        std::lock_guard<std::mutex> lock(chunk_mutex);
        if (idx >= chunk_phase.size()){
            return;
        }
        chunk_phase[idx] = 2; // done
    }

    py::list chunkStatuses() const{
        py::list out;
        std::unique_lock<std::mutex> lock(chunk_mutex, std::try_to_lock);
        if (!lock.owns_lock()){
            return out;
        }
        for (size_t i = 0; i < chunk_attempts.size(); ++i){
            py::dict d;
            d["chunk"] = i;
            d["attempts"] = chunk_attempts[i];
            d["regions"] = chunk_regions[i];
            d["remaining"] = chunk_remaining[i];
            d["fail_rate"] = chunk_fail_rate[i];
            d["fail_threshold"] = chunk_fail_threshold[i];
            d["phase"] = static_cast<int>(chunk_phase[i]);
            out.append(std::move(d));
        }
        return out;
    }
};

void validate_coords(const py::buffer_info& buf, const char* name);
void validate_normals(const py::buffer_info& coords, const py::buffer_info& normals);
void load_coords_from_numpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& coords,
    PointCloud& pcd
);
void load_normals_from_numpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& normals,
    PointCloud& pcd
);
RegionGrowingParams parse_region_params(const py::dict& d);

class SeededGrowerSession {
public:
    SeededGrowerSession(
        py::array_t<float, py::array::c_style | py::array::forcecast> coords,
        py::object normals_or_none,
        const std::string& normal_mode,
        const py::dict& params_dict
    ){
        auto coords_buf = coords.request();
        validate_coords(coords_buf, "coords");
        load_coords_from_numpy(coords, pcd_);
        params_ = parse_region_params(params_dict);

        const bool mode_compute = (normal_mode == "compute");
        const bool mode_provided = (normal_mode == "use_provided" || normal_mode == "use_existing");
        if (!mode_compute && !mode_provided){
            throw std::runtime_error("normal_mode must be 'compute' or 'use_provided'");
        }
        if (mode_provided){
            if (normals_or_none.is_none()){
                throw std::runtime_error("normal_mode='use_provided' requires normals");
            }
            auto normals = normals_or_none.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            auto normals_buf = normals.request();
            validate_normals(coords_buf, normals_buf);
            load_normals_from_numpy(normals, pcd_);
        }

        kdtree_ = std::make_unique<KDTreeType>(3, pcd_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        kdtree_->buildIndex();

        if (mode_compute){
            pcd_.normals.assign(pcd_.coords.size(), Point{0.0f, 0.0f, 1.0f});
            std::unordered_set<size_t> all_indices;
            all_indices.reserve(pcd_.coords.size());
            for (size_t i = 0; i < pcd_.coords.size(); ++i){
                all_indices.insert(i);
            }
            py::gil_scoped_release release;
            computeNormals(pcd_.coords, pcd_.normals, all_indices, *kdtree_, params_.epsilon);
        }
    }

    py::tuple grow(size_t seed_index){
        if (seed_index >= pcd_.coords.size()){
            throw std::runtime_error("seed_index out of range");
        }
        Region reg;
        {
            py::gil_scoped_release release;
            reg = growRegionFromSeed(pcd_, *kdtree_, seed_index, nullptr, params_);
        }

        std::vector<int32_t> indices;
        indices.reserve(reg.indices.size());
        for (const auto idx : reg.indices){
            indices.push_back(static_cast<int32_t>(idx));
        }
        std::sort(indices.begin(), indices.end());

        py::array_t<int32_t> out({static_cast<py::ssize_t>(indices.size())});
        if (!indices.empty()){
            auto out_buf = out.request();
            std::memcpy(out_buf.ptr, indices.data(), indices.size() * sizeof(int32_t));
        }

        py::dict stats;
        stats["seed_index"] = seed_index;
        stats["num_points"] = indices.size();
        stats["accepted"] = !indices.empty();
        return py::make_tuple(out, stats);
    }

private:
    PointCloud pcd_;
    std::unique_ptr<KDTreeType> kdtree_;
    RegionGrowingParams params_;
};

struct ChunkingConfig {
    bool enabled = true;
    std::string mode = "auto";
    size_t target_points_per_chunk = 250000;
    size_t chunk_x = 1;
    size_t chunk_y = 1;
    size_t chunk_z = 1;
    float overlap_factor = 3.0f;
};

struct MergeConfig {
    bool enabled = true;
    float angle_deg = 5.0f;
    float distance_factor = 3.0f;
};

struct ChunkData {
    size_t id = 0;
    std::vector<size_t> primary_indices;
    std::unordered_set<size_t> task_indices;
};

struct PairHash {
    size_t operator()(const std::pair<size_t, size_t>& p) const noexcept{
        return std::hash<size_t>{}(p.first) ^ (std::hash<size_t>{}(p.second) << 1);
    }
};

template <typename T>
T dict_get(const py::dict& d, const char* key, const T& default_value){
    if (!d || !d.contains(key)){
        return default_value;
    }
    return d[key].cast<T>();
}

void validate_coords(const py::buffer_info& buf, const char* name);
void validate_normals(const py::buffer_info& coords, const py::buffer_info& normals);
void load_coords_from_numpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& coords,
    PointCloud& pcd
);
void load_normals_from_numpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& normals,
    PointCloud& pcd
);
RegionGrowingParams parse_region_params(const py::dict& d);

void validate_coords(const py::buffer_info& buf, const char* name){
    if (buf.ndim != 2 || buf.shape[1] != 3){
        throw std::runtime_error(std::string(name) + " must be a (N,3) float array");
    }
}

void validate_normals(const py::buffer_info& coords, const py::buffer_info& normals){
    validate_coords(normals, "normals");
    if (coords.shape[0] != normals.shape[0]){
        throw std::runtime_error("normals must have the same number of rows as coords");
    }
}

float median_inplace(std::vector<float>& values){
    if (values.empty()){
        return 0.0f;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    float med = values[mid];
    if (values.size() % 2 == 0){
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        med = 0.5f * (med + values[mid - 1]);
    }
    return med;
}

float median_size_t(const std::vector<size_t>& values){
    if (values.empty()){
        return 0.0f;
    }
    std::vector<float> as_float;
    as_float.reserve(values.size());
    for (const auto v : values){
        as_float.push_back(static_cast<float>(v));
    }
    return median_inplace(as_float);
}

void load_coords_from_numpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& coords,
    PointCloud& pcd
){
    auto buf = coords.request();
    validate_coords(buf, "coords");
    const auto* ptr = static_cast<const float*>(buf.ptr);
    const size_t n = static_cast<size_t>(buf.shape[0]);
    pcd.coords.resize(n);
    for (size_t i = 0; i < n; ++i){
        pcd.coords[i] = Point{ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]};
    }
}

void load_normals_from_numpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& normals,
    PointCloud& pcd
){
    auto buf = normals.request();
    validate_coords(buf, "normals");
    const auto* ptr = static_cast<const float*>(buf.ptr);
    const size_t n = static_cast<size_t>(buf.shape[0]);
    pcd.normals.resize(n);
    for (size_t i = 0; i < n; ++i){
        pcd.normals[i] = Point{ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]};
    }
}

RegionGrowingParams parse_region_params(const py::dict& d){
    RegionGrowingParams params;
    params.epsilon = dict_get<float>(d, "epsilon", params.epsilon);
    params.refit_multiplier = dict_get<float>(d, "refit_multiplier", params.refit_multiplier);
    params.epsilon_multiplier = dict_get<float>(d, "epsilon_multiplier", params.epsilon_multiplier);
    params.epsilon_multiplier_average = dict_get<float>(d, "epsilon_multiplier_average", params.epsilon_multiplier_average);
    params.search_radius_approx = dict_get<float>(d, "search_radius_approx", params.search_radius_approx);
    params.min_points_in_region = dict_get<size_t>(d, "tau", params.min_points_in_region);
    params.first_refit = dict_get<size_t>(d, "first_refit", params.first_refit);
    const float alpha_deg = dict_get<float>(d, "alpha_deg", params.alpha * 180.0f / static_cast<float>(PI));
    params.alpha = alpha_deg * static_cast<float>(PI) / 180.0f;
    params.stop_confidence = dict_get<float>(d, "confidence", params.stop_confidence);
    params.max_dist_from_cent = dict_get<float>(d, "max_dist_from_cent", params.max_dist_from_cent);
    params.oriented_normals = dict_get<bool>(d, "oriented_normals", params.oriented_normals);
    params.verbose = dict_get<bool>(d, "verbose", false);
    params.perform_cca = dict_get<bool>(d, "perform_cca", params.perform_cca);
    params.enable_seed_gating = dict_get<bool>(d, "enable_seed_gating", params.enable_seed_gating);
    params.seed_min_neighbors = dict_get<size_t>(d, "seed_min_neighbors", params.seed_min_neighbors);
    params.seed_planarity_min = dict_get<float>(d, "seed_planarity_min", params.seed_planarity_min);
    params.seed_scattering_max = dict_get<float>(d, "seed_scattering_max", params.seed_scattering_max);
    params.failrate_window = dict_get<size_t>(d, "failrate_window", params.failrate_window);
    params.failrate_threshold = dict_get<float>(d, "failrate_threshold", params.failrate_threshold);
    params.stop_confidence = std::clamp(params.stop_confidence, 1e-4f, 0.999999f);
    params.min_points_in_region = std::max<size_t>(3, params.min_points_in_region);
    params.epsilon = std::max(1e-6f, params.epsilon);
    params.epsilon_multiplier = std::max(1e-3f, params.epsilon_multiplier);
    params.max_dist_from_cent = std::max(1e-3f, params.max_dist_from_cent);
    params.seed_min_neighbors = std::max<size_t>(3, params.seed_min_neighbors);
    params.seed_planarity_min = std::clamp(params.seed_planarity_min, 0.0f, 1.0f);
    params.seed_scattering_max = std::clamp(params.seed_scattering_max, 0.0f, 1.0f);
    params.failrate_window = std::max<size_t>(8, params.failrate_window);
    params.failrate_threshold = std::clamp(params.failrate_threshold, 0.5f, 0.999f);
    return params;
}

ChunkingConfig parse_chunking_config(const py::dict& d){
    ChunkingConfig cfg;
    cfg.enabled = dict_get<bool>(d, "enabled", cfg.enabled);
    cfg.mode = dict_get<std::string>(d, "mode", cfg.mode);
    cfg.target_points_per_chunk = dict_get<size_t>(d, "target_points_per_chunk", cfg.target_points_per_chunk);
    cfg.chunk_x = dict_get<size_t>(d, "chunk_x", cfg.chunk_x);
    cfg.chunk_y = dict_get<size_t>(d, "chunk_y", cfg.chunk_y);
    cfg.chunk_z = dict_get<size_t>(d, "chunk_z", cfg.chunk_z);
    cfg.overlap_factor = dict_get<float>(d, "overlap_factor", cfg.overlap_factor);
    cfg.target_points_per_chunk = std::max<size_t>(1, cfg.target_points_per_chunk);
    cfg.chunk_x = std::max<size_t>(1, cfg.chunk_x);
    cfg.chunk_y = std::max<size_t>(1, cfg.chunk_y);
    cfg.chunk_z = std::max<size_t>(1, cfg.chunk_z);
    cfg.overlap_factor = std::max(2.5f, cfg.overlap_factor);
    return cfg;
}

MergeConfig parse_merge_config(const py::dict& d){
    MergeConfig cfg;
    cfg.enabled = dict_get<bool>(d, "enabled", cfg.enabled);
    cfg.angle_deg = dict_get<float>(d, "angle_deg", cfg.angle_deg);
    cfg.distance_factor = dict_get<float>(d, "distance_factor", cfg.distance_factor);
    cfg.angle_deg = std::clamp(cfg.angle_deg, 0.0f, 90.0f);
    cfg.distance_factor = std::max(0.0f, cfg.distance_factor);
    return cfg;
}

std::array<size_t, 3> auto_chunk_dims(size_t n_points, size_t target_points, const Eigen::Vector3f& aabb_size){
    const size_t desired_chunks = std::max<size_t>(1, static_cast<size_t>(
        std::ceil(static_cast<double>(n_points) / static_cast<double>(target_points))
    ));
    std::array<size_t, 3> dims{1, 1, 1};
    std::array<float, 3> extent = {
        std::max(1e-3f, aabb_size[0]),
        std::max(1e-3f, aabb_size[1]),
        std::max(1e-3f, aabb_size[2]),
    };
    while (dims[0] * dims[1] * dims[2] < desired_chunks){
        size_t axis = 0;
        float best_score = extent[0] / static_cast<float>(dims[0]);
        for (size_t a = 1; a < 3; ++a){
            const float score = extent[a] / static_cast<float>(dims[a]);
            if (score > best_score){
                best_score = score;
                axis = a;
            }
        }
        ++dims[axis];
    }
    return dims;
}

int clamp_int(int value, int lo, int hi){
    return std::max(lo, std::min(hi, value));
}

std::vector<ChunkData> build_chunks(
    const PointCloud& pcd,
    const ChunkingConfig& cfg,
    float epsilon
){
    const size_t n = pcd.coords.size();
    if (n == 0){
        return {};
    }

    size_t chunk_x = 1;
    size_t chunk_y = 1;
    size_t chunk_z = 1;

    const auto [aabb_min, aabb_max] = getPcdAabb(pcd);
    const Eigen::Vector3f aabb_size = aabb_max - aabb_min;
    if (cfg.enabled){
        if (cfg.mode == "explicit"){
            chunk_x = cfg.chunk_x;
            chunk_y = cfg.chunk_y;
            chunk_z = cfg.chunk_z;
        } else {
            const auto dims = auto_chunk_dims(n, cfg.target_points_per_chunk, aabb_size);
            chunk_x = dims[0];
            chunk_y = dims[1];
            chunk_z = dims[2];
        }
    }

    const size_t total_chunks = chunk_x * chunk_y * chunk_z;
    const Eigen::Vector3f chunk_size = {
        aabb_size[0] / static_cast<float>(chunk_x),
        aabb_size[1] / static_cast<float>(chunk_y),
        aabb_size[2] / static_cast<float>(chunk_z),
    };
    const float overlap_halo = cfg.overlap_factor * epsilon;

    std::unordered_map<size_t, ChunkData> chunks;
    chunks.reserve(total_chunks);

    for (size_t i = 0; i < n; ++i){
        const Eigen::Vector3f rel = pcd.coords[i] - aabb_min;
        Eigen::Vector3i owner = rel.cwiseQuotient(chunk_size).cast<int>();
        owner[0] = clamp_int(owner[0], 0, static_cast<int>(chunk_x) - 1);
        owner[1] = clamp_int(owner[1], 0, static_cast<int>(chunk_y) - 1);
        owner[2] = clamp_int(owner[2], 0, static_cast<int>(chunk_z) - 1);
        const size_t owner_id = flattenIndex(
            static_cast<size_t>(owner[0]),
            chunk_x,
            static_cast<size_t>(owner[1]),
            chunk_y,
            static_cast<size_t>(owner[2]),
            chunk_z
        );
        auto& owner_chunk = chunks[owner_id];
        owner_chunk.id = owner_id;
        owner_chunk.primary_indices.push_back(i);

        const Eigen::Vector3f rel_min = (pcd.coords[i] - Eigen::Vector3f::Constant(overlap_halo)) - aabb_min;
        const Eigen::Vector3f rel_max = (pcd.coords[i] + Eigen::Vector3f::Constant(overlap_halo)) - aabb_min;
        Eigen::Vector3i min_cell = rel_min.cwiseQuotient(chunk_size).array().floor().matrix().cast<int>();
        Eigen::Vector3i max_cell = rel_max.cwiseQuotient(chunk_size).array().floor().matrix().cast<int>();
        min_cell[0] = clamp_int(min_cell[0], 0, static_cast<int>(chunk_x) - 1);
        min_cell[1] = clamp_int(min_cell[1], 0, static_cast<int>(chunk_y) - 1);
        min_cell[2] = clamp_int(min_cell[2], 0, static_cast<int>(chunk_z) - 1);
        max_cell[0] = clamp_int(max_cell[0], 0, static_cast<int>(chunk_x) - 1);
        max_cell[1] = clamp_int(max_cell[1], 0, static_cast<int>(chunk_y) - 1);
        max_cell[2] = clamp_int(max_cell[2], 0, static_cast<int>(chunk_z) - 1);

        for (int cx = min_cell[0]; cx <= max_cell[0]; ++cx){
            for (int cy = min_cell[1]; cy <= max_cell[1]; ++cy){
                for (int cz = min_cell[2]; cz <= max_cell[2]; ++cz){
                    const size_t chunk_id = flattenIndex(
                        static_cast<size_t>(cx),
                        chunk_x,
                        static_cast<size_t>(cy),
                        chunk_y,
                        static_cast<size_t>(cz),
                        chunk_z
                    );
                    auto& chunk = chunks[chunk_id];
                    chunk.id = chunk_id;
                    chunk.task_indices.insert(i);
                }
            }
        }
    }

    std::vector<ChunkData> out;
    out.reserve(chunks.size());
    for (auto& kv : chunks){
        out.push_back(std::move(kv.second));
    }
    std::sort(out.begin(), out.end(), [](const ChunkData& a, const ChunkData& b){
        return a.id < b.id;
    });
    return out;
}

void refit_region(Region& reg, const PointCloud& pcd){
    if (reg.indices.empty()){
        reg.centroid = Point{0.0f, 0.0f, 0.0f};
        reg.normal = Point{0.0f, 0.0f, 1.0f};
        return;
    }
    Eigen::Matrix<float, Eigen::Dynamic, 3> reg_coords(reg.indices.size(), 3);
    Point centroid = {0.0f, 0.0f, 0.0f};
    size_t idx = 0;
    for (const auto p_idx : reg.indices){
        reg_coords.row(idx) = pcd.coords[p_idx];
        centroid += pcd.coords[p_idx];
        ++idx;
    }
    centroid /= static_cast<float>(reg.indices.size());
    reg_coords.rowwise() -= centroid.transpose();
    Eigen::Matrix3f covariance = reg_coords.transpose() * reg_coords;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    reg.normal = solver.eigenvectors().col(0);
    reg.centroid = centroid;
}

void cleanup_unassigned_components_and_points(
    PointCloud& pcd,
    KDTreeType& kdtree,
    std::unordered_map<size_t, size_t>& point_to_region,
    std::unordered_set<size_t>& unassigned,
    std::vector<Region>& regions,
    const RegionGrowingParams& params,
    std::vector<int32_t>* labels,
    bool assign_fallback_components
){
    if (unassigned.empty() || regions.empty()){
        if (unassigned.empty() || !assign_fallback_components || labels == nullptr){
            return;
        }

        auto comps_only = unionFindCCA(pcd, unassigned, params.epsilon);
        int32_t next_instance = 0;
        for (const auto& comp : comps_only){
            for (const auto idx : comp){
                (*labels)[idx] = next_instance;
            }
            ++next_instance;
        }
        return;
    }

    auto components = unionFindCCA(pcd, unassigned, params.epsilon);
    std::unordered_set<size_t> changed_regions;

    for (const auto& comp_inds : components){
        if (comp_inds.empty()){
            continue;
        }

        std::unordered_set<size_t> candidate_regions;
        for (const auto idx : comp_inds){
            IndicesDistType indices_dist;
            RadiusSetType result_set(params.epsilon, indices_dist);
            kdtree.findNeighbors(
                result_set,
                pcd.coords[idx].data(),
                nanoflann::SearchParameters(params.search_radius_approx, true)
            );
            for (const auto& i_d : indices_dist){
                auto map_it = point_to_region.find(i_d.first);
                if (map_it != point_to_region.end()){
                    candidate_regions.insert(map_it->second);
                }
            }
        }
        if (candidate_regions.empty()){
            continue;
        }

        // Assign each point in this component independently to its closest
        // candidate plane (subject to plane-distance threshold).
        for (const auto idx : comp_inds){
            bool found = false;
            float best_dist = std::numeric_limits<float>::max();
            size_t best_region = 0;
            for (const auto cand_reg : candidate_regions){
                const Region& reg = regions[cand_reg];
                const float dist_to_plane = std::abs((pcd.coords[idx] - reg.centroid).dot(reg.normal));
                if (!found || dist_to_plane < best_dist){
                    found = true;
                    best_dist = dist_to_plane;
                    best_region = cand_reg;
                }
            }
            if (found && best_dist <= params.epsilon_multiplier * params.epsilon){
                if (labels != nullptr){
                    (*labels)[idx] = static_cast<int32_t>(best_region);
                }
                point_to_region[idx] = best_region;
                regions[best_region].indices.insert(idx);
                unassigned.erase(idx);
                changed_regions.insert(best_region);
            }
        }
    }

    for (const auto reg_id : changed_regions){
        refit_region(regions[reg_id], pcd);
    }

    for (auto it = unassigned.begin(); it != unassigned.end();){
        IndicesDistType indices_dist;
        RadiusSetType result_set(params.epsilon, indices_dist);
        kdtree.findNeighbors(
            result_set,
            pcd.coords[*it].data(),
            nanoflann::SearchParameters(params.search_radius_approx, true)
        );

        bool found = false;
        float min_dist_to_plane = std::numeric_limits<float>::max();
        float min_criterion = std::numeric_limits<float>::max();
        size_t best_region = 0;
        const float eps_safe = std::max(1e-8f, params.epsilon);
        const float eps_plane = std::max(1e-8f, params.epsilon * params.epsilon_multiplier);

        for (const auto& i_d : indices_dist){
            auto map_it = point_to_region.find(i_d.first);
            if (map_it == point_to_region.end()){
                continue;
            }
            const size_t reg_id = map_it->second;
            const Region& reg = regions[reg_id];
            const float dist_to_plane = std::abs((pcd.coords[*it] - reg.centroid).dot(reg.normal));
            const float criterion =
                (static_cast<float>(i_d.second) / eps_safe) + (dist_to_plane / eps_plane);
            if (!found || criterion < min_criterion){
                found = true;
                min_criterion = criterion;
                min_dist_to_plane = dist_to_plane;
                best_region = reg_id;
            }
        }

        if (found && min_dist_to_plane <= params.epsilon_multiplier * params.epsilon){
            if (labels != nullptr){
                (*labels)[*it] = static_cast<int32_t>(best_region);
            }
            point_to_region[*it] = best_region;
            regions[best_region].indices.insert(*it);
            it = unassigned.erase(it);
        } else {
            ++it;
        }
    }

    // Fallback: convert still-unassigned connected components into their own
    // instance labels so no point remains at -1.
    if (assign_fallback_components && labels != nullptr && !unassigned.empty()){
        auto leftover_components = unionFindCCA(pcd, unassigned, params.epsilon);
        int32_t next_instance = 0;
        for (const auto v : *labels){
            if (v >= next_instance){
                next_instance = v + 1;
            }
        }
        for (const auto& comp : leftover_components){
            if (comp.empty()){
                continue;
            }
            for (const auto idx : comp){
                (*labels)[idx] = next_instance;
            }
            ++next_instance;
        }
    }
}

void assign_small_components_then_points(
    PointCloud& pcd,
    KDTreeType& kdtree,
    std::vector<int32_t>& labels,
    std::vector<Region>& regions,
    const RegionGrowingParams& params
){
    std::unordered_map<size_t, size_t> point_to_region;
    std::unordered_set<size_t> unassigned;
    for (size_t i = 0; i < labels.size(); ++i){
        if (labels[i] >= 0){
            point_to_region[i] = static_cast<size_t>(labels[i]);
        } else {
            unassigned.insert(i);
        }
    }
    cleanup_unassigned_components_and_points(
        pcd,
        kdtree,
        point_to_region,
        unassigned,
        regions,
        params,
        &labels,
        true
    );
}

void assign_local_leftovers(
    PointCloud& pcd,
    KDTreeType& kdtree,
    regionGrowing_returnType& result,
    const RegionGrowingParams& params
){
    cleanup_unassigned_components_and_points(
        pcd,
        kdtree,
        result.pcd_to_reg_idxmap,
        result.unassigned,
        result.regions,
        params,
        nullptr,
        false
    );
}

py::tuple estimate_parameters_impl(
    py::array_t<float, py::array::c_style | py::array::forcecast> coords,
    size_t sample_size,
    uint32_t seed
){
    auto coords_buf = coords.request();
    validate_coords(coords_buf, "coords");

    const size_t n_points = static_cast<size_t>(coords_buf.shape[0]);
    PointCloud pcd;
    load_coords_from_numpy(coords, pcd);

    if (n_points == 0){
        py::dict diagnostics;
        diagnostics["sample_size_used"] = 0;
        diagnostics["tau0"] = 0;
        diagnostics["base_spacing"] = 0.0f;
        diagnostics["estimate_seconds"] = 0.0;
        return py::make_tuple(0.03f, static_cast<size_t>(80), 29.0f, diagnostics);
    }

    const auto t0 = std::chrono::steady_clock::now();
    KDTreeType kdtree(3, pcd, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();

    sample_size = std::max<size_t>(1, std::min(sample_size, n_points));
    std::vector<size_t> sampled_indices;
    sampled_indices.reserve(sample_size);
    for (size_t i = 0; i < n_points; ++i){
        sampled_indices.push_back(i);
    }
    std::mt19937 rng(seed);
    std::shuffle(sampled_indices.begin(), sampled_indices.end(), rng);
    sampled_indices.resize(sample_size);

    std::vector<float> kth_neighbor_dist;
    kth_neighbor_dist.reserve(sample_size);
    const size_t k_spacing = 8;
    std::vector<uint32_t> knn_indices(k_spacing);
    std::vector<float> knn_dists(k_spacing);
    for (const auto idx : sampled_indices){
        const size_t found = kdtree.knnSearch(
            pcd.coords[idx].data(),
            static_cast<size_t>(k_spacing),
            knn_indices.data(),
            knn_dists.data()
        );
        if (found >= k_spacing){
            kth_neighbor_dist.push_back(std::sqrt(knn_dists[k_spacing - 1]));
        }
    }
    float base_spacing = median_inplace(kth_neighbor_dist);
    if (base_spacing <= 0.0f){
        base_spacing = 0.05f;
    }

    const std::array<float, 8> radius_mult = {1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 10.0f};
    std::vector<float> candidate_radii;
    candidate_radii.reserve(radius_mult.size());
    for (const auto m : radius_mult){
        candidate_radii.push_back(std::max(1e-4f, m * base_spacing));
    }

    std::vector<float> epsilon_candidates;
    epsilon_candidates.reserve(sample_size);
    for (const auto idx : sampled_indices){
        float chosen = candidate_radii.back();
        for (const auto r : candidate_radii){
            IndicesDistType neighb_inds_dist;
            RadiusSetType result_set(r, neighb_inds_dist);
            kdtree.findNeighbors(
                result_set,
                pcd.coords[idx].data(),
                nanoflann::SearchParameters(0.0f, false)
            );
            if (neighb_inds_dist.size() < 6){
                continue;
            }
            Eigen::Matrix<float, Eigen::Dynamic, 3> neighbors(neighb_inds_dist.size(), 3);
            for (size_t k = 0; k < neighb_inds_dist.size(); ++k){
                neighbors.row(k) = pcd.coords[neighb_inds_dist[k].first] - pcd.coords[idx];
            }
            Eigen::Matrix3f covariance = neighbors.transpose() * neighbors;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
            const auto evals = solver.eigenvalues();
            const float l1 = std::max(evals[0], 1e-12f);
            const float l2 = evals[1];
            if ((l2 / l1) >= 3.0f){
                chosen = r;
                break;
            }
        }
        epsilon_candidates.push_back(chosen);
    }
    float epsilon = std::max(1e-4f, median_inplace(epsilon_candidates));

    std::vector<size_t> neighb_counts;
    neighb_counts.reserve(sample_size);
    for (const auto idx : sampled_indices){
        IndicesDistType neighb_inds_dist;
        RadiusSetType result_set(epsilon, neighb_inds_dist);
        kdtree.findNeighbors(
            result_set,
            pcd.coords[idx].data(),
            nanoflann::SearchParameters(0.0f, false)
        );
        neighb_counts.push_back(neighb_inds_dist.size());
    }
    const float tau0 = std::max(3.0f, median_size_t(neighb_counts));
    const size_t tau = static_cast<size_t>(std::round(tau0));

    const float ratio = static_cast<float>(tau) / std::max(1.0f, tau0);
    const float cos_arg = std::clamp(1.0f - ratio / 8.0f, -1.0f, 1.0f);
    const float alpha_deg = std::acos(cos_arg) * 180.0f / static_cast<float>(PI);

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    py::dict diagnostics;
    diagnostics["sample_size_used"] = sample_size;
    diagnostics["tau0"] = tau0;
    diagnostics["base_spacing"] = base_spacing;
    diagnostics["estimate_seconds"] = elapsed_seconds;
    diagnostics["seed"] = seed;
    return py::make_tuple(epsilon, tau, alpha_deg, diagnostics);
}

py::tuple segment_planar_regions_impl(
    py::array_t<float, py::array::c_style | py::array::forcecast> coords,
    py::object normals_or_none,
    const std::string& normal_mode,
    const py::dict& params_dict,
    const py::dict& chunking_dict,
    const py::dict& merge_dict,
    ProgressState* progress
){
    auto coords_buf = coords.request();
    validate_coords(coords_buf, "coords");
    const size_t n_points = static_cast<size_t>(coords_buf.shape[0]);

    PointCloud pcd;
    load_coords_from_numpy(coords, pcd);

    RegionGrowingParams rg_params = parse_region_params(params_dict);
    ChunkingConfig chunking_cfg = parse_chunking_config(chunking_dict);
    MergeConfig merge_cfg = parse_merge_config(merge_dict);
    const bool local_reassign_enabled =
        dict_get<bool>(params_dict, "local_reassign_enabled", true);
    const bool global_reassign_enabled = params_dict.contains("global_reassign_enabled")
        ? dict_get<bool>(params_dict, "global_reassign_enabled", true)
        : dict_get<bool>(params_dict, "refine_unassigned", true);

    KDTreeType kdtree(3, pcd, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();

    std::vector<int32_t> labels(n_points, -1);

    const auto t0 = std::chrono::steady_clock::now();
    const bool mode_compute = (normal_mode == "compute");
    const bool mode_provided = (normal_mode == "use_provided" || normal_mode == "use_existing");
    if (!mode_compute && !mode_provided){
        throw std::runtime_error("normal_mode must be 'compute' or 'use_provided'");
    }

    if (!mode_compute) {
        std::cout << "[region_growing] Stage 1/6: using provided normals..." << std::endl;
        if (normals_or_none.is_none()){
            throw std::runtime_error("normal_mode='use_provided' requires normals");
        }
        auto normals = normals_or_none.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        auto normals_buf = normals.request();
        validate_normals(coords_buf, normals_buf);
        load_normals_from_numpy(normals, pcd);
    }

    std::vector<ChunkData> chunks;
    size_t region_count_before_merge = 0;
    std::vector<Region> merged_regions;
    size_t unassigned_count = 0;

    {
    py::gil_scoped_release release;

    if (mode_compute){
        std::cout << "[region_growing] Stage 1/6: normals will be computed inside chunk workers..." << std::endl;
    }

    chunks = build_chunks(pcd, chunking_cfg, rg_params.epsilon);
    std::cout
        << "[region_growing] Stage 2/6: chunking prepared with "
        << chunks.size() << " chunk(s)"
        << (chunking_cfg.enabled ? "" : " (chunking disabled)")
        << std::endl;
    if (chunks.empty()){
        chunks.push_back(ChunkData{0, {}, {}});
        chunks[0].primary_indices.reserve(n_points);
        for (size_t i = 0; i < n_points; ++i){
            chunks[0].primary_indices.push_back(i);
            chunks[0].task_indices.insert(i);
        }
    }

    if (progress != nullptr){
        progress->reset(static_cast<int64_t>(chunks.size() + 4));
        progress->initChunks(chunks.size());
    }

    std::vector<regionGrowing_returnType> chunk_results(chunks.size());
    std::exception_ptr thread_error = nullptr;
    std::mutex error_mutex;
    std::mutex print_mutex;
    std::atomic<size_t> chunks_done{0};
    std::vector<std::thread> threads;
    threads.reserve(chunks.size());

    for (size_t chunk_i = 0; chunk_i < chunks.size(); ++chunk_i){
        threads.emplace_back([&, chunk_i]{
            try {
                std::unordered_map<size_t, Point> local_normals;
                const std::unordered_map<size_t, Point>* normal_override = nullptr;
                if (mode_compute){
                    computeNormalsSparse(
                        pcd.coords,
                        local_normals,
                        chunks[chunk_i].task_indices,
                        kdtree,
                        rg_params.epsilon
                    );
                    normal_override = &local_normals;
                }
                chunk_results[chunk_i] = regionGrowing(
                    pcd,
                    kdtree,
                    &chunks[chunk_i].task_indices,
                    rg_params,
                    [&](const RegionGrowingRuntime& rt){
                        if (progress == nullptr){
                            return;
                        }
                        const bool should_emit = rt.done || (rt.attempts % 25 == 0);
                        if (!should_emit){
                            return;
                        }
                        progress->updateChunk(
                            chunk_i,
                            static_cast<int64_t>(rt.attempts),
                            static_cast<int64_t>(rt.regions_found),
                            static_cast<int64_t>(rt.unassigned_remaining),
                            static_cast<double>(rt.rolling_fail_rate),
                            static_cast<double>(rt.rolling_fail_threshold),
                            rt.done
                        );
                    },
                    normal_override
                );
                if (local_reassign_enabled){
                    assign_local_leftovers(
                        pcd,
                        kdtree,
                        chunk_results[chunk_i],
                        rg_params
                    );
                }
                const size_t finished = chunks_done.fetch_add(1, std::memory_order_relaxed) + 1;
                const size_t every = std::max<size_t>(1, chunks.size() / 10);
                if (finished == 1 || finished == chunks.size() || (finished % every) == 0){
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout
                        << "[region_growing] Stage 3/6: segmented chunks "
                        << finished << "/" << chunks.size()
                        << std::endl;
                }
                if (progress != nullptr){
                    progress->completeChunk(chunk_i);
                }
                if (progress != nullptr){
                    progress->done.fetch_add(1, std::memory_order_relaxed);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (thread_error == nullptr){
                    thread_error = std::current_exception();
                }
            }
        });
    }
    for (auto& t : threads){
        t.join();
    }
    if (thread_error != nullptr){
        std::rethrow_exception(thread_error);
    }

    std::vector<Region> global_regions;
    std::vector<std::vector<size_t>> point_region_memberships(n_points);
    region_count_before_merge = 0;

    for (size_t chunk_i = 0; chunk_i < chunks.size(); ++chunk_i){
        auto& result = chunk_results[chunk_i];
        std::vector<size_t> local_to_global(result.regions.size(), 0);
        for (size_t local = 0; local < result.regions.size(); ++local){
            local_to_global[local] = global_regions.size();
            global_regions.push_back(std::move(result.regions[local]));
        }
        region_count_before_merge += local_to_global.size();

        for (const auto& kv : result.pcd_to_reg_idxmap){
            const size_t pt_idx = kv.first;
            const size_t global_id = local_to_global[kv.second];
            point_region_memberships[pt_idx].push_back(global_id);
        }
        for (const auto pt_idx : chunks[chunk_i].primary_indices){
            auto map_it = result.pcd_to_reg_idxmap.find(pt_idx);
            if (map_it != result.pcd_to_reg_idxmap.end()){
                labels[pt_idx] = static_cast<int32_t>(local_to_global[map_it->second]);
            } else {
                labels[pt_idx] = -1;
            }
        }
    }

    if (progress != nullptr){
        progress->done.fetch_add(1, std::memory_order_relaxed);
    }

    std::cout
        << "[region_growing] Stage 4/6: reconciliation "
        << (merge_cfg.enabled ? "enabled" : "disabled")
        << std::endl;
    std::vector<size_t> parent(global_regions.size());
    std::vector<size_t> rank(global_regions.size(), 0);
    for (size_t i = 0; i < parent.size(); ++i){
        parent[i] = i;
    }

    auto uf_find = [&](size_t x){
        size_t root = x;
        while (parent[root] != root){
            root = parent[root];
        }
        while (parent[x] != x){
            const size_t px = parent[x];
            parent[x] = root;
            x = px;
        }
        return root;
    };

    auto uf_union = [&](size_t a, size_t b){
        a = uf_find(a);
        b = uf_find(b);
        if (a == b){
            return;
        }
        if (rank[a] < rank[b]){
            std::swap(a, b);
        }
        parent[b] = a;
        if (rank[a] == rank[b]){
            ++rank[a];
        }
    };

    if (merge_cfg.enabled){
        std::unordered_map<std::pair<size_t, size_t>, size_t, PairHash> pair_votes;
        for (auto& memberships : point_region_memberships){
            if (memberships.size() < 2){
                continue;
            }
            std::sort(memberships.begin(), memberships.end());
            memberships.erase(std::unique(memberships.begin(), memberships.end()), memberships.end());
            for (size_t i = 0; i < memberships.size(); ++i){
                for (size_t j = i + 1; j < memberships.size(); ++j){
                    pair_votes[{memberships[i], memberships[j]}] += 1;
                }
            }
        }
        const float merge_dist_thresh = merge_cfg.distance_factor * rg_params.epsilon;
        for (const auto& kv : pair_votes){
            const size_t a = kv.first.first;
            const size_t b = kv.first.second;
            const Region& ra = global_regions[a];
            const Region& rb = global_regions[b];
            const float dot = std::clamp(
                std::abs(ra.normal.normalized().dot(rb.normal.normalized())),
                -1.0f,
                1.0f
            );
            const float angle_deg = std::acos(dot) * 180.0f / static_cast<float>(PI);
            if (angle_deg > merge_cfg.angle_deg){
                continue;
            }
            const float d_ab = std::abs((rb.centroid - ra.centroid).dot(ra.normal.normalized()));
            const float d_ba = std::abs((ra.centroid - rb.centroid).dot(rb.normal.normalized()));
            if (d_ab <= merge_dist_thresh && d_ba <= merge_dist_thresh){
                uf_union(a, b);
            }
        }
    }

    std::unordered_map<size_t, size_t> root_to_compact;
    for (size_t i = 0; i < global_regions.size(); ++i){
        const size_t root = uf_find(i);
        if (root_to_compact.find(root) == root_to_compact.end()){
            root_to_compact[root] = root_to_compact.size();
        }
    }

    merged_regions = std::vector<Region>(root_to_compact.size());
    for (size_t i = 0; i < n_points; ++i){
        if (labels[i] < 0){
            continue;
        }
        const size_t root = uf_find(static_cast<size_t>(labels[i]));
        const size_t compact = root_to_compact[root];
        labels[i] = static_cast<int32_t>(compact);
        merged_regions[compact].indices.insert(i);
    }
    for (auto& reg : merged_regions){
        refit_region(reg, pcd);
    }

    if (progress != nullptr){
        progress->done.fetch_add(1, std::memory_order_relaxed);
    }

    std::cout
        << "[region_growing] Stage 5/6: global leftover reassignment "
        << (global_reassign_enabled ? "enabled" : "disabled")
        << std::endl;
    if (global_reassign_enabled){
        assign_small_components_then_points(pcd, kdtree, labels, merged_regions, rg_params);
    }

    if (progress != nullptr){
        progress->done.fetch_add(1, std::memory_order_relaxed);
        progress->done.store(progress->total.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    unassigned_count = 0;
    for (const auto label : labels){
        if (label < 0){
            ++unassigned_count;
        }
    }
    } // gil_scoped_release

    py::dict stats;
    py::array_t<int32_t> labels_arr({static_cast<py::ssize_t>(n_points)});
    auto out_buf = labels_arr.request();
    std::memcpy(out_buf.ptr, labels.data(), labels.size() * sizeof(int32_t));

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    stats["num_points"] = n_points;
    stats["num_chunks"] = chunks.size();
    stats["num_regions_pre_merge"] = region_count_before_merge;
    stats["num_regions_post_merge"] = merged_regions.size();
    stats["num_unassigned"] = unassigned_count;
    stats["elapsed_seconds"] = elapsed_seconds;
    stats["cancelled"] = progress != nullptr && progress->isCancelled();
    std::cout << "[region_growing] Stage 6/6: done in " << elapsed_seconds << "s" << std::endl;

    return py::make_tuple(labels_arr, stats);
}

} // namespace

PYBIND11_MODULE(region_growing, m){
    m.doc() = "Planar region growing module";

    py::class_<ProgressState>(m, "Progress", py::module_local())
        .def(py::init<>())
        .def("reset", &ProgressState::reset, py::arg("total"))
        .def("request_cancel", &ProgressState::requestCancel)
        .def("cancelled", &ProgressState::isCancelled)
        .def("done", &ProgressState::completed)
        .def("total", &ProgressState::totalCount)
        .def("chunk_statuses", &ProgressState::chunkStatuses);

    py::class_<SeededGrowerSession>(m, "SeededGrower")
        .def(
            py::init<
                py::array_t<float, py::array::c_style | py::array::forcecast>,
                py::object,
                const std::string&,
                const py::dict&
            >(),
            py::arg("coords"),
            py::arg("normals") = py::none(),
            py::kw_only(),
            py::arg("normal_mode") = "compute",
            py::arg("params") = py::dict()
        )
        .def("grow", &SeededGrowerSession::grow, py::arg("seed_index"));

    m.def(
        "estimate_parameters",
        &estimate_parameters_impl,
        py::arg("coords"),
        py::kw_only(),
        py::arg("sample_size") = 50000,
        py::arg("seed") = 0
    );

    m.def(
        "segment_planar_regions",
        &segment_planar_regions_impl,
        py::arg("coords"),
        py::arg("normals") = py::none(),
        py::kw_only(),
        py::arg("normal_mode") = "compute",
        py::arg("params") = py::dict(),
        py::arg("chunking") = py::dict(),
        py::arg("merge") = py::dict(),
        py::arg("progress") = py::none()
    );
}
