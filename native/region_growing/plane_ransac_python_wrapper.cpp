#include "plane_ransac.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

struct ProgressState {
    std::atomic<int64_t> done{0};
    std::atomic<int64_t> total{0};
    std::atomic<bool> cancel{false};
    mutable std::mutex mutex;
    std::string stage = "Idle";
    int64_t planes_found = 0;
    int64_t active_points_remaining = 0;
    int64_t current_best_support = 0;

    void reset(int64_t total_count){
        done.store(0, std::memory_order_relaxed);
        total.store(total_count, std::memory_order_relaxed);
        cancel.store(false, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mutex);
        stage = "Idle";
        planes_found = 0;
        active_points_remaining = total_count;
        current_best_support = 0;
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

    void updateStatus(const PlaneRansacRuntime& runtime){
        done.store(static_cast<int64_t>(runtime.assigned_points), std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mutex);
        stage = runtime.stage;
        planes_found = static_cast<int64_t>(runtime.planes_found);
        active_points_remaining = static_cast<int64_t>(runtime.active_points_remaining);
        current_best_support = static_cast<int64_t>(runtime.current_best_support);
    }

    std::string stageText() const{
        std::lock_guard<std::mutex> lock(mutex);
        return stage;
    }

    int64_t planesFound() const{
        std::lock_guard<std::mutex> lock(mutex);
        return planes_found;
    }

    int64_t activePointsRemaining() const{
        std::lock_guard<std::mutex> lock(mutex);
        return active_points_remaining;
    }

    int64_t currentBestSupport() const{
        std::lock_guard<std::mutex> lock(mutex);
        return current_best_support;
    }
};

template <typename T>
T dict_get(const py::dict& d, const char* key, const T& default_value){
    if (!d || !d.contains(key)){
        return default_value;
    }
    return d[key].cast<T>();
}

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

PlaneRansacParams parse_params(const py::dict& d){
    PlaneRansacParams params;
    params.epsilon = dict_get<float>(d, "epsilon", params.epsilon);
    params.min_points = dict_get<size_t>(d, "min_points", params.min_points);
    params.normal_threshold_deg = dict_get<float>(d, "normal_threshold_deg", params.normal_threshold_deg);
    params.cluster_epsilon = dict_get<float>(d, "cluster_epsilon", params.cluster_epsilon);
    params.probability = dict_get<float>(d, "probability", params.probability);
    params.max_iterations_per_plane = dict_get<size_t>(d, "max_iterations_per_plane", params.max_iterations_per_plane);
    params.seed = dict_get<uint32_t>(d, "seed", params.seed);
    params.epsilon = std::max(1e-6f, params.epsilon);
    params.min_points = std::max<size_t>(3, params.min_points);
    params.normal_threshold_deg = std::clamp(params.normal_threshold_deg, 0.0f, 89.9f);
    params.probability = std::clamp(params.probability, 1e-6f, 0.999999f);
    params.max_iterations_per_plane = std::max<size_t>(100, params.max_iterations_per_plane);
    return params;
}

py::tuple segment_planes_impl(
    py::array_t<float, py::array::c_style | py::array::forcecast> coords,
    py::object normals_or_none,
    const std::string& normal_mode,
    const py::dict& params_dict,
    ProgressState* progress
){
    auto coords_buf = coords.request();
    validate_coords(coords_buf, "coords");
    const size_t n_points = static_cast<size_t>(coords_buf.shape[0]);

    PointCloud pcd;
    load_coords_from_numpy(coords, pcd);
    const PlaneRansacParams params = parse_params(params_dict);

    if (progress != nullptr){
        progress->reset(static_cast<int64_t>(n_points));
    }

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
        load_normals_from_numpy(normals, pcd);
    }

    PlaneRansacKDTree kdtree(3, pcd, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();

    PlaneRansacResult result;
    const auto t0 = std::chrono::steady_clock::now();
    {
        py::gil_scoped_release release;

        if (mode_compute){
            std::cout << "[plane_ransac] Computing normals..." << std::endl;
            if (progress != nullptr){
                PlaneRansacRuntime rt;
                rt.stage = "Computing normals...";
                rt.active_points_remaining = n_points;
                progress->updateStatus(rt);
            }
            compute_plane_ransac_normals(pcd.coords, pcd.normals, kdtree, params.epsilon);
        } else {
            std::cout << "[plane_ransac] Using provided normals..." << std::endl;
            if (progress != nullptr){
                PlaneRansacRuntime rt;
                rt.stage = "Using provided normals...";
                rt.active_points_remaining = n_points;
                progress->updateStatus(rt);
            }
        }

        result = segment_planes_ransac(
            pcd,
            kdtree,
            params,
            [&](const PlaneRansacRuntime& runtime){
                if (progress != nullptr){
                    progress->updateStatus(runtime);
                    if (runtime.done){
                        progress->done.store(
                            progress->total.load(std::memory_order_relaxed),
                            std::memory_order_relaxed
                        );
                    }
                }
                return progress == nullptr || !progress->isCancelled();
            }
        );
    }

    py::array_t<int32_t> labels_arr({static_cast<py::ssize_t>(n_points)});
    auto out_buf = labels_arr.request();
    std::memcpy(out_buf.ptr, result.labels.data(), result.labels.size() * sizeof(int32_t));

    size_t unassigned_count = 0;
    for (const auto label : result.labels){
        if (label < 0){
            ++unassigned_count;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

    py::dict stats;
    stats["num_points"] = n_points;
    stats["num_planes"] = result.planes.size();
    stats["num_unassigned"] = unassigned_count;
    stats["elapsed_seconds"] = elapsed_seconds;
    stats["cancelled"] = result.cancelled || (progress != nullptr && progress->isCancelled());

    py::list planes_out;
    for (const auto& plane : result.planes){
        py::dict p;
        p["centroid"] = py::make_tuple(plane.centroid[0], plane.centroid[1], plane.centroid[2]);
        p["normal"] = py::make_tuple(plane.normal[0], plane.normal[1], plane.normal[2]);
        p["d"] = plane.d;
        p["size"] = plane.indices.size();
        planes_out.append(std::move(p));
    }
    stats["planes"] = std::move(planes_out);
    std::cout << "[plane_ransac] Done in " << elapsed_seconds << "s" << std::endl;

    return py::make_tuple(labels_arr, stats);
}

} // namespace

PYBIND11_MODULE(plane_ransac, m){
    m.doc() = "Plane-only RANSAC segmentation";

    py::class_<ProgressState>(m, "Progress", py::module_local())
        .def(py::init<>())
        .def("reset", &ProgressState::reset, py::arg("total"))
        .def("request_cancel", &ProgressState::requestCancel)
        .def("cancelled", &ProgressState::isCancelled)
        .def("done", &ProgressState::completed)
        .def("total", &ProgressState::totalCount)
        .def("stage", &ProgressState::stageText)
        .def("planes_found", &ProgressState::planesFound)
        .def("active_points_remaining", &ProgressState::activePointsRemaining)
        .def("current_best_support", &ProgressState::currentBestSupport);

    m.def(
        "segment_planes",
        &segment_planes_impl,
        py::arg("coords"),
        py::arg("normals") = py::none(),
        py::kw_only(),
        py::arg("normal_mode") = "compute",
        py::arg("params") = py::dict(),
        py::arg("progress") = py::none()
    );
}
