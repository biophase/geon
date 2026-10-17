#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace py = pybind11;

// Read coordinates in place, including strided arrays. Projection arithmetic
// stays double precision, matching the previous NumPy projection matrix.
template <typename T>
py::array_t<int64_t> select_polygon(
    py::array_t<T, 0> points,
    py::array_t<int64_t, py::array::forcecast> candidates,
    py::array_t<double, py::array::forcecast> matrix,
    int width, int height,
    py::array_t<double, py::array::forcecast> polygon, bool keep_inside) {
    if (points.ndim() != 2 || points.shape(1) != 3)
        throw py::value_error("points must have shape (N, 3)");
    if (candidates.ndim() != 1)
        throw py::value_error("candidates must be one-dimensional");
    if (matrix.ndim() != 2 || matrix.shape(0) != 4 || matrix.shape(1) != 4)
        throw py::value_error("matrix must have shape (4, 4)");
    if (polygon.ndim() != 2 || polygon.shape(1) != 2 || polygon.shape(0) < 3)
        throw py::value_error("polygon must have shape (M, 2), M >= 3");
    if (width <= 0 || height <= 0)
        throw py::value_error("viewport dimensions must be positive");
    auto pts = points.template unchecked<2>();
    auto ids = candidates.unchecked<1>();
    auto m = matrix.unchecked<2>();
    auto poly = polygon.unchecked<2>();
    double xmin = poly(0, 0), xmax = xmin, ymin = poly(0, 1), ymax = ymin;
    for (py::ssize_t j = 0; j < poly.shape(0); ++j) {
        if (!std::isfinite(poly(j, 0)) || !std::isfinite(poly(j, 1)))
            throw py::value_error("polygon must be finite");
        xmin = std::min(xmin, poly(j, 0)); xmax = std::max(xmax, poly(j, 0));
        ymin = std::min(ymin, poly(j, 1)); ymax = std::max(ymax, poly(j, 1));
    }
    std::vector<int64_t> selected;
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < ids.shape(0); ++i) {
            const int64_t id = ids(i);

            if (id < 0 || id >= pts.shape(0))
                throw py::value_error("candidate index out of bounds");
            const double x = pts(id, 0), y = pts(id, 1), z = pts(id, 2);
            const double w = x*m(3,0) + y*m(3,1) + z*m(3,2) + m(3,3);
            bool inside = false;

            if (w > 0) {
                const double sx = ((x*m(0,0) + y*m(0,1) + z*m(0,2) + m(0,3))/w + 1)/2*width;
                const double sy = ((x*m(1,0) + y*m(1,1) + z*m(1,2) + m(1,3))/w + 1)/2*height;
                if (sx >= xmin && sx <= xmax && sy >= ymin && sy <= ymax) {
                    for (py::ssize_t j = 0; j < poly.shape(0); ++j) {
                        const auto k = (j + 1) % poly.shape(0);
                        const double x0 = poly(j,0), y0 = poly(j,1);
                        const double x1 = poly(k,0), y1 = poly(k,1);

                        // Preserve the existing ray-crossing boundary convention.
                        if (((y0 > sy) != (y1 > sy)) &&
                            sx < (x1-x0)*(sy-y0)/(y1-y0+1e-12)+x0)
                            inside = !inside;
                    }
                }
            }
            if (inside == keep_inside) selected.push_back(id);
        }
    }
    py::array_t<int64_t> result(selected.size());
    std::copy(selected.begin(), selected.end(), result.mutable_data());
    return result;
}

PYBIND11_MODULE(lasso, m) {
    m.doc() = "Fused screen projection and polygon selection";
    m.def("select_polygon", &select_polygon<float>, py::arg("points").noconvert(),
          py::arg("candidates"), py::arg("matrix"), py::arg("width"), py::arg("height"),
          py::arg("polygon"), py::arg("keep_inside") = true);
    m.def("select_polygon", &select_polygon<double>, py::arg("points").noconvert(),
          py::arg("candidates"), py::arg("matrix"), py::arg("width"), py::arg("height"),
          py::arg("polygon"), py::arg("keep_inside") = true);
}
