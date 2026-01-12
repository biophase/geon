#include "features.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

struct VoxelHashIndex {
    VoxelHash map;
};

void validate_coords(const py::buffer_info& buf, const char* name){
    if (buf.ndim != 2 || buf.shape[1] != 3){
        throw std::runtime_error(std::string(name) + " must be a (N,3) float array");
    }
}

Point parse_query(const py::buffer_info& buf){
    if (buf.ndim == 1 && buf.shape[0] == 3){
        const auto* ptr = static_cast<float*>(buf.ptr);
        return Point{ptr[0], ptr[1], ptr[2]};
    }
    if (buf.ndim == 2 && buf.shape[0] == 1 && buf.shape[1] == 3){
        const auto* ptr = static_cast<float*>(buf.ptr);
        return Point{ptr[0], ptr[1], ptr[2]};
    }
    throw std::runtime_error("query must be a (3,) or (1,3) float array");
}

} // namespace

PYBIND11_MODULE(features, m){
    m.doc() = "Point cloud feature utilities";

    py::class_<VoxelHashIndex>(m, "VoxelHash")
        .def(py::init<>())
        .def("__len__", [](const VoxelHashIndex& self){ return self.map.size(); });

    py::class_<ProgressState>(m, "Progress")
        .def(py::init<>())
        .def("reset", &ProgressState::reset, py::arg("total"))
        .def("request_cancel", &ProgressState::requestCancel)
        .def("cancelled", &ProgressState::isCancelled)
        .def("done", &ProgressState::completed)
        .def("total", &ProgressState::totalCount);

    m.def(
        "voxel_key",
        &voxelKey,
        py::arg("x"),
        py::arg("y"),
        py::arg("z"),
        py::arg("inv_s")
    );

    m.def(
        "compute_voxel_hash",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> positive_coords,
           float inv_s){
            auto buf = positive_coords.request();
            validate_coords(buf, "positive_coords");

            const auto n_rows = static_cast<Eigen::Index>(buf.shape[0]);
            MapMatNx3fRM coords_map(static_cast<float*>(buf.ptr), n_rows, 3);

            float inv_s_local = inv_s;
            VoxelHashIndex voxel_hash;
            {
                py::gil_scoped_release release;
                voxel_hash.map = computeVoxelHash(coords_map, inv_s_local);
            }
            return voxel_hash;
        },
        py::arg("positive_coords"),
        py::arg("inv_s")
    );

    m.def(
        "get_neighbor_inds_radius",
        [](float radius,
           py::array_t<float, py::array::c_style | py::array::forcecast> query,
           float voxel_size,
           const VoxelHashIndex& voxel_hash,
           py::array_t<float, py::array::c_style | py::array::forcecast> positive_coords){
            auto query_buf = query.request();
            Point query_point = parse_query(query_buf);

            auto coords_buf = positive_coords.request();
            validate_coords(coords_buf, "positive_coords");
            MapMatNx3fRM coords_map(
                static_cast<float*>(coords_buf.ptr),
                static_cast<Eigen::Index>(coords_buf.shape[0]),
                3
            );

            std::vector<uint32_t> neighbors;
            {
                py::gil_scoped_release release;
                neighbors = getNeighborIndsRadius(
                    radius,
                    query_point,
                    voxel_size,
                    voxel_hash.map,
                    coords_map
                );
            }

            py::array_t<uint32_t> out({static_cast<py::ssize_t>(neighbors.size())});
            auto out_buf = out.request();
            auto* out_ptr = static_cast<uint32_t*>(out_buf.ptr);
            std::copy(neighbors.begin(), neighbors.end(), out_ptr);
            return out;
        },
        py::arg("radius"),
        py::arg("query"),
        py::arg("voxel_size"),
        py::arg("voxel_hash"),
        py::arg("positive_coords")
    );

    m.def(
        "compute_pcd_features",
        [](float radius,
           float voxel_size,
           py::array_t<float, py::array::c_style | py::array::forcecast> positive_coords,
           const VoxelHashIndex& voxel_hash,
           ProgressState* progress){
            auto coords_buf = positive_coords.request();
            validate_coords(coords_buf, "positive_coords");

            const auto n_rows = static_cast<Eigen::Index>(coords_buf.shape[0]);
            MapMatNx3fRM coords_map(static_cast<float*>(coords_buf.ptr), n_rows, 3);

            py::array eigenvalues(py::dtype::of<float>(), {
                static_cast<py::ssize_t>(coords_buf.shape[0]),
                static_cast<py::ssize_t>(3)
            });
            py::array normals(py::dtype::of<float>(), {
                static_cast<py::ssize_t>(coords_buf.shape[0]),
                static_cast<py::ssize_t>(3)
            });

            auto ev_buf = eigenvalues.request();
            auto n_buf = normals.request();
            MapMatNx3fRM ev_map(static_cast<float*>(ev_buf.ptr), n_rows, 3);
            MapMatNx3fRM n_map(static_cast<float*>(n_buf.ptr), n_rows, 3);

            {
                py::gil_scoped_release release;
                computePcdFeatures(
                    radius,
                    voxel_size,
                    coords_map,
                    voxel_hash.map,
                    ev_map,
                    n_map,
                    progress
                );
            }

            return py::make_tuple(eigenvalues, normals);
        },
        py::arg("radius"),
        py::arg("voxel_size"),
        py::arg("positive_coords"),
        py::arg("voxel_hash"),
        py::arg("progress") = py::none()
    );
}
