#include "subsampling.h"

#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void validate_coords(const py::buffer_info& buf){
    if (buf.ndim != 2 || buf.shape[1] != 3){
        throw std::runtime_error("coords must be a (N,3) float array");
    }
}

} // namespace

PYBIND11_MODULE(subsampling, m){
    m.doc() = "Spatial subsampling utilities";

    m.def(
        "spatial_subsample_mask",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> coords,
           float min_distance){
            auto buf = coords.request();
            validate_coords(buf);

            const auto n_rows = static_cast<Eigen::Index>(buf.shape[0]);
            SubsampleMapMatNx3fRM coords_map(static_cast<float*>(buf.ptr), n_rows, 3);

            std::vector<uint8_t> mask;
            {
                py::gil_scoped_release release;
                mask = spatialSubsampleMask(coords_map, min_distance);
            }

            py::array_t<bool> out({static_cast<py::ssize_t>(mask.size())});
            auto out_buf = out.request();
            auto* out_ptr = static_cast<bool*>(out_buf.ptr);
            for (size_t i = 0; i < mask.size(); ++i){
                out_ptr[i] = (mask[i] != 0);
            }
            return out;
        },
        py::arg("coords"),
        py::arg("min_distance")
    );
}
