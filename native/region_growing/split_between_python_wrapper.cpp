#include "split_between.h"

#include <cstring>
#include <stdexcept>
#include <unordered_set>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

PointCloud load_coords(const py::array_t<float, py::array::c_style | py::array::forcecast>& coords){
    const auto buffer = coords.request();
    if (buffer.ndim != 2 || buffer.shape[1] != 3){
        throw std::runtime_error("coords must be a (N,3) float array");
    }
    const auto* data = static_cast<const float*>(buffer.ptr);
    PointCloud pcd;
    pcd.coords.resize(static_cast<size_t>(buffer.shape[0]));
    for (size_t i = 0; i < pcd.coords.size(); ++i){
        pcd.coords[i] = Point{data[3 * i], data[3 * i + 1], data[3 * i + 2]};
    }
    return pcd;
}

std::vector<size_t> load_indices(const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& array,
                                 size_t point_count, const char* name){
    const auto buffer = array.request();
    if (buffer.ndim != 1){ throw std::runtime_error(std::string(name) + " must be one-dimensional"); }
    const auto* data = static_cast<const int64_t*>(buffer.ptr);
    std::vector<size_t> result;
    result.reserve(static_cast<size_t>(buffer.shape[0]));
    std::unordered_set<int64_t> seen;
    for (py::ssize_t i = 0; i < buffer.shape[0]; ++i){
        if (data[i] < 0 || static_cast<size_t>(data[i]) >= point_count){
            throw std::runtime_error(std::string(name) + " contains an out-of-range index");
        }
        if (!seen.insert(data[i]).second){
            throw std::runtime_error(std::string(name) + " contains duplicate indices");
        }
        result.push_back(static_cast<size_t>(data[i]));
    }
    return result;
}

} // namespace

PYBIND11_MODULE(split_between, module){
    module.doc() = "Assign working points between two reference point sets";
    auto progress_type = py::class_<SplitBetweenProgressState>(module, "_SplitBetweenProgress", py::module_local())
        .def(py::init<>())
        .def("request_cancel", &SplitBetweenProgressState::requestCancel)
        .def("cancelled", &SplitBetweenProgressState::isCancelled)
        .def("done", &SplitBetweenProgressState::completed)
        .def("total", &SplitBetweenProgressState::totalCount)
        .def("stage", &SplitBetweenProgressState::stageText);
    module.attr("Progress") = progress_type;
    module.def("split_between", [](
        py::array_t<float, py::array::c_style | py::array::forcecast> coords,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> working,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> set_a,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> set_b,
        const std::string& condition,
        py::object progress_object){
        PointCloud pcd = load_coords(coords);
        auto work_indices = load_indices(working, pcd.coords.size(), "working_indices");
        auto a_indices = load_indices(set_a, pcd.coords.size(), "a_indices");
        auto b_indices = load_indices(set_b, pcd.coords.size(), "b_indices");
        std::unordered_set<size_t> occupied(work_indices.begin(), work_indices.end());
        for (const size_t idx : a_indices){ if (!occupied.insert(idx).second) throw std::runtime_error("sets must be disjoint"); }
        for (const size_t idx : b_indices){ if (!occupied.insert(idx).second) throw std::runtime_error("sets must be disjoint"); }
        SplitBetweenCondition mode;
        if (condition == "plane_distance") mode = SplitBetweenCondition::PlaneDistance;
        else if (condition == "nearest_neighbor") mode = SplitBetweenCondition::NearestNeighbor;
        else throw std::runtime_error("condition must be 'plane_distance' or 'nearest_neighbor'");
        SplitBetweenProgressState* progress = progress_object.is_none()
            ? nullptr : progress_object.cast<SplitBetweenProgressState*>();
        SplitBetweenResult result;
        { py::gil_scoped_release release;
          result = split_between_impl(pcd, work_indices, a_indices, b_indices, mode, progress); }
        py::array_t<uint8_t> assignment({static_cast<py::ssize_t>(result.assignment.size())});
        std::memcpy(assignment.mutable_data(), result.assignment.data(), result.assignment.size());
        py::dict stats;
        stats["assigned_to_a"] = result.assigned_to_a;
        stats["assigned_to_b"] = result.assigned_to_b;
        stats["elapsed_seconds"] = result.elapsed_seconds;
        stats["cancelled"] = result.cancelled;
        return py::make_tuple(assignment, stats);
    }, py::arg("coords"), py::arg("working_indices"), py::arg("a_indices"),
       py::arg("b_indices"), py::arg("condition"), py::arg("progress") = py::none());
}
