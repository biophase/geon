#include "superpoints.h"

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

void validate_features(const py::buffer_info& coords, const py::buffer_info& features){
    if (features.ndim != 2){
        throw std::runtime_error("features must be a (N,F) float array");
    }
    if (features.shape[0] != coords.shape[0]){
        throw std::runtime_error("features must have the same number of rows as coords");
    }
}

template <typename T>
T kwarg_or(const py::kwargs& kwargs, const char* key, const T& default_value){
    if (!kwargs || !kwargs.contains(key)){
        return default_value;
    }
    return kwargs[key].cast<T>();
}

PointCloud load_point_cloud(
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

std::vector<float> load_extra_features(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& features,
    int32_t& feature_dim_out
){
    auto buf = features.request();
    feature_dim_out = static_cast<int32_t>(buf.shape[1]);
    const auto* ptr = static_cast<const float*>(buf.ptr);
    return std::vector<float>(ptr, ptr + (buf.shape[0] * buf.shape[1]));
}

} // namespace

PYBIND11_MODULE(superpoints, m){
    m.doc() = "Superpoint segmentation via parallel cut pursuit";

    // Keep the registered pybind type name unique across extension modules.
    auto progress_type = py::class_<SuperpointProgressState>(
        m,
        "_SuperpointsProgress",
        py::module_local()
    )
        .def(py::init<>())
        .def("reset", &SuperpointProgressState::reset, py::arg("total"))
        .def("request_cancel", &SuperpointProgressState::requestCancel)
        .def("cancelled", &SuperpointProgressState::isCancelled)
        .def("done", &SuperpointProgressState::completed)
        .def("total", &SuperpointProgressState::totalCount)
        .def("stage", &SuperpointProgressState::stageText);
    m.attr("Progress") = progress_type;

    m.def(
        "segment_superpoints",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> coords,
           py::object features_or_none,
           py::kwargs kwargs){
            auto coords_buf = coords.request();
            validate_coords(coords_buf, "coords");

            PointCloud pcd = load_point_cloud(coords);
            std::vector<float> extra_features;
            int32_t extra_feature_dim = 0;
            if (!features_or_none.is_none()){
                auto features = features_or_none.cast<
                    py::array_t<float, py::array::c_style | py::array::forcecast>
                >();
                auto features_buf = features.request();
                validate_features(coords_buf, features_buf);
                extra_features = load_extra_features(features, extra_feature_dim);
            }

            SuperpointParams params;
            params.k_neighbors = std::max(1, kwarg_or<int32_t>(kwargs, "k_neighbors", params.k_neighbors));
            params.regularization = std::max(1e-6f, kwarg_or<float>(kwargs, "regularization", params.regularization));
            params.spatial_weight = std::max(1e-6f, kwarg_or<float>(kwargs, "spatial_weight", params.spatial_weight));
            params.cutoff = std::max(1, kwarg_or<int32_t>(kwargs, "cutoff", params.cutoff));
            params.iterations = std::max(1, kwarg_or<int32_t>(kwargs, "iterations", params.iterations));
            params.parallel = kwarg_or<bool>(kwargs, "parallel", params.parallel);
            params.verbose = kwarg_or<bool>(kwargs, "verbose", params.verbose);

            SuperpointProgressState* progress = nullptr;
            if (kwargs && kwargs.contains("progress") && !kwargs["progress"].is_none()){
                progress = kwargs["progress"].cast<SuperpointProgressState*>();
            }

            SuperpointResult result;
            {
                py::gil_scoped_release release;
                result = segment_superpoints_impl(
                    pcd,
                    extra_features,
                    extra_feature_dim,
                    params,
                    progress
                );
            }

            py::array_t<int32_t> labels_arr({static_cast<py::ssize_t>(result.labels.size())});
            std::memcpy(
                labels_arr.mutable_data(),
                result.labels.data(),
                result.labels.size() * sizeof(int32_t)
            );

            py::dict stats;
            stats["num_points"] = result.stats.num_points;
            stats["num_superpoints"] = result.stats.num_superpoints;
            stats["feature_dim"] = result.stats.feature_dim;
            stats["num_edges"] = result.stats.num_edges;
            stats["elapsed_seconds"] = result.stats.elapsed_seconds;
            stats["cancelled"] = result.stats.cancelled;
            return py::make_tuple(labels_arr, stats);
        },
        py::arg("coords"),
        py::arg("features") = py::none()
    );
}
