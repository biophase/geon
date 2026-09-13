#include "corner_cleanup.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void validate_coords(const py::buffer_info& buffer){
    if (buffer.ndim != 2 || buffer.shape[1] != 3){
        throw std::runtime_error("coords must be a (N,3) float array");
    }
}

void validate_labels(const py::buffer_info& labels, const py::buffer_info& coords){
    if (!(labels.ndim == 1 || (labels.ndim == 2 && labels.shape[1] == 1))){
        throw std::runtime_error("labels must be a (N,) or (N,1) int array");
    }
    if (labels.shape[0] != coords.shape[0]){
        throw std::runtime_error("labels length must match coords row count");
    }
}

template <typename T>
T dict_get(const py::dict& values, const char* key, const T& default_value){
    if (!values || !values.contains(key)){
        return default_value;
    }
    return values[key].cast<T>();
}

PointCloud load_coords(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& coords
){
    const auto buffer = coords.request();
    const auto* data = static_cast<const float*>(buffer.ptr);
    PointCloud pcd;
    pcd.coords.resize(static_cast<size_t>(buffer.shape[0]));
    for (size_t i = 0; i < pcd.coords.size(); ++i){
        pcd.coords[i] = Point{data[i * 3], data[i * 3 + 1], data[i * 3 + 2]};
    }
    return pcd;
}

std::vector<int32_t> load_labels(
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& labels
){
    const auto buffer = labels.request();
    const auto* data = static_cast<const int32_t*>(buffer.ptr);
    std::vector<int32_t> result(static_cast<size_t>(buffer.shape[0]));
    std::memcpy(result.data(), data, result.size() * sizeof(int32_t));
    return result;
}

CornerCleanupParams parse_params(const py::dict& values){
    CornerCleanupParams params;
    params.epsilon = std::max(1e-6f, dict_get<float>(values, "epsilon", params.epsilon));
    params.neighbor_radius_factor = std::max(
        1e-6f,
        dict_get<float>(values, "neighbor_radius_factor", params.neighbor_radius_factor)
    );
    params.slenderness_threshold = std::clamp(
        dict_get<float>(values, "slenderness_threshold", params.slenderness_threshold), 0.0f, 1.0f);
    params.absolute_width_factor = std::max(
        0.0f, dict_get<float>(values, "absolute_width_factor", params.absolute_width_factor));
    params.relative_width_factor = std::clamp(
        dict_get<float>(values, "relative_width_factor", params.relative_width_factor), 0.0f, 1.0f);
    params.dual_support_threshold = std::clamp(
        dict_get<float>(values, "dual_support_threshold", params.dual_support_threshold),
        0.0f,
        1.0f
    );
    params.min_corner_size = std::max(
        3,
        dict_get<int32_t>(values, "min_corner_size", params.min_corner_size)
    );
    params.min_planar_size = std::max(
        3,
        dict_get<int32_t>(values, "min_planar_size", params.min_planar_size)
    );
    return params;
}

} // namespace

PYBIND11_MODULE(corner_cleanup, module){
    module.doc() = "Cleanup elongated corner regions between planar instances";

    auto progress_type = py::class_<CornerCleanupProgressState>(
        module,
        "_CornerCleanupProgress",
        py::module_local()
    )
        .def(py::init<>())
        .def("reset", &CornerCleanupProgressState::reset, py::arg("total"))
        .def("request_cancel", &CornerCleanupProgressState::requestCancel)
        .def("cancelled", &CornerCleanupProgressState::isCancelled)
        .def("done", &CornerCleanupProgressState::completed)
        .def("total", &CornerCleanupProgressState::totalCount)
        .def("stage", &CornerCleanupProgressState::stageText);
    module.attr("Progress") = progress_type;

    module.def(
        "cleanup_corner_regions",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> coords,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> labels,
           const py::dict& params_dict,
           py::object progress_object){
            const auto coords_buffer = coords.request();
            const auto labels_buffer = labels.request();
            validate_coords(coords_buffer);
            validate_labels(labels_buffer, coords_buffer);

            PointCloud pcd = load_coords(coords);
            std::vector<int32_t> label_values = load_labels(labels);
            const CornerCleanupParams params = parse_params(params_dict);
            CornerCleanupProgressState* progress = nullptr;
            if (!progress_object.is_none()){
                progress = progress_object.cast<CornerCleanupProgressState*>();
            }

            CornerCleanupResult result;
            {
                py::gil_scoped_release release;
                result = cleanup_corner_regions_impl(pcd, label_values, params, progress);
            }

            py::array_t<int32_t> output({static_cast<py::ssize_t>(result.labels.size())});
            std::memcpy(
                output.mutable_data(),
                result.labels.data(),
                result.labels.size() * sizeof(int32_t)
            );
            py::dict stats;
            stats["num_points"] = result.stats.num_points;
            stats["num_input_regions"] = result.stats.num_input_regions;
            stats["num_destination_regions"] = result.stats.num_destination_regions;
            stats["neighbor_radius"] = result.stats.neighbor_radius;
            stats["num_corner_candidates"] = result.stats.num_corner_candidates;
            stats["num_rejected_slenderness"] = result.stats.num_rejected_slenderness;
            stats["num_rejected_absolute_width"] = result.stats.num_rejected_absolute_width;
            stats["num_rejected_relative_width"] = result.stats.num_rejected_relative_width;
            stats["num_rejected_planar_neighbors"] = result.stats.num_rejected_planar_neighbors;
            stats["num_rejected_dual_support"] = result.stats.num_rejected_dual_support;
            stats["num_corner_regions_cleaned"] = result.stats.num_corner_regions_cleaned;
            stats["num_corner_points_reassigned"] = result.stats.num_corner_points_reassigned;
            stats["elapsed_seconds"] = result.stats.elapsed_seconds;
            stats["cancelled"] = result.stats.cancelled;
            return py::make_tuple(output, stats);
        },
        py::arg("coords"),
        py::arg("labels"),
        py::arg("params") = py::dict(),
        py::arg("progress") = py::none()
    );
}
