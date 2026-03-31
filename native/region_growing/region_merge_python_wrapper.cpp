#include "region_merge.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

void validate_coords(const py::buffer_info& buf, const char* name){
    if (buf.ndim != 2 || buf.shape[1] != 3){
        throw std::runtime_error(std::string(name) + " must be a (N,3) float array");
    }
}

void validate_labels(const py::buffer_info& buf, const py::buffer_info& coords){
    if (!((buf.ndim == 1) || (buf.ndim == 2 && buf.shape[1] == 1))){
        throw std::runtime_error("labels must be a (N,) or (N,1) int array");
    }
    const py::ssize_t n = buf.ndim == 1 ? buf.shape[0] : buf.shape[0];
    if (n != coords.shape[0]){
        throw std::runtime_error("labels length must match coords row count");
    }
}

template <typename T>
T dict_get(const py::dict& d, const char* key, const T& default_value){
    if (!d || !d.contains(key)){
        return default_value;
    }
    return d[key].cast<T>();
}

PointCloud load_coords(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& coords
){
    auto buf = coords.request();
    validate_coords(buf, "coords");
    const auto* ptr = static_cast<const float*>(buf.ptr);
    const size_t n = static_cast<size_t>(buf.shape[0]);
    PointCloud pcd;
    pcd.coords.resize(n);
    for (size_t i = 0; i < n; ++i){
        pcd.coords[i] = Point{ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]};
    }
    return pcd;
}

std::vector<int32_t> load_labels(
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& labels
){
    auto buf = labels.request();
    const auto* ptr = static_cast<const int32_t*>(buf.ptr);
    const size_t n = static_cast<size_t>(buf.shape[0]);
    std::vector<int32_t> out(n);
    if (buf.ndim == 1){
        std::memcpy(out.data(), ptr, n * sizeof(int32_t));
    } else {
        for (size_t i = 0; i < n; ++i){
            out[i] = ptr[i];
        }
    }
    return out;
}

RegionMergeParams parse_params(const py::dict& d){
    RegionMergeParams params;
    params.neighbor_radius = std::max(1e-6f, dict_get<float>(d, "neighbor_radius", params.neighbor_radius));
    params.min_contact_points = std::max(1, dict_get<int32_t>(d, "min_contact_points", params.min_contact_points));
    params.planarity_threshold = std::clamp(dict_get<float>(d, "planarity_threshold", params.planarity_threshold), 0.0f, 1.0f);
    params.normal_angle_deg = std::clamp(dict_get<float>(d, "normal_angle_deg", params.normal_angle_deg), 0.0f, 90.0f);
    params.plane_distance_threshold = std::max(0.0f, dict_get<float>(d, "plane_distance_threshold", params.plane_distance_threshold));
    params.min_region_size = std::max(1, dict_get<int32_t>(d, "min_region_size", params.min_region_size));
    return params;
}

} // namespace

PYBIND11_MODULE(region_merge, m){
    m.doc() = "Planar region merging for instance fields";

    // Keep the registered pybind type name unique across extension modules.
    auto progress_type = py::class_<RegionMergeProgressState>(
        m,
        "_RegionMergeProgress",
        py::module_local()
    )
        .def(py::init<>())
        .def("reset", &RegionMergeProgressState::reset, py::arg("total"))
        .def("request_cancel", &RegionMergeProgressState::requestCancel)
        .def("cancelled", &RegionMergeProgressState::isCancelled)
        .def("done", &RegionMergeProgressState::completed)
        .def("total", &RegionMergeProgressState::totalCount)
        .def("stage", &RegionMergeProgressState::stageText);
    m.attr("Progress") = progress_type;

    m.def(
        "merge_planar_regions",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> coords,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> labels,
           const py::dict& params_dict,
           py::object progress_obj){
            auto coords_buf = coords.request();
            auto labels_buf = labels.request();
            validate_coords(coords_buf, "coords");
            validate_labels(labels_buf, coords_buf);

            PointCloud pcd = load_coords(coords);
            std::vector<int32_t> labels_vec = load_labels(labels);
            RegionMergeParams params = parse_params(params_dict);
            RegionMergeProgressState* progress = nullptr;
            if (!progress_obj.is_none()){
                progress = progress_obj.cast<RegionMergeProgressState*>();
            }

            RegionMergeResult result;
            {
                py::gil_scoped_release release;
                result = merge_planar_regions_impl(pcd, labels_vec, params, progress);
            }

            py::array_t<int32_t> labels_arr({static_cast<py::ssize_t>(result.labels.size())});
            std::memcpy(
                labels_arr.mutable_data(),
                result.labels.data(),
                result.labels.size() * sizeof(int32_t)
            );

            py::dict stats;
            stats["num_points"] = result.stats.num_points;
            stats["num_input_regions"] = result.stats.num_input_regions;
            stats["num_planar_regions"] = result.stats.num_planar_regions;
            stats["num_adjacency_pairs"] = result.stats.num_adjacency_pairs;
            stats["num_merge_candidates"] = result.stats.num_merge_candidates;
            stats["num_output_regions"] = result.stats.num_output_regions;
            stats["elapsed_seconds"] = result.stats.elapsed_seconds;
            stats["cancelled"] = result.stats.cancelled;
            return py::make_tuple(labels_arr, stats);
        },
        py::arg("coords"),
        py::arg("labels"),
        py::arg("params") = py::dict(),
        py::arg("progress") = py::none()
    );
}
