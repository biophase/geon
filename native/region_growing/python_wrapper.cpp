#include "types.h"
#include "rgrow.h"
#include "io.h"

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

// py::array_t<float> pcd_get_coords(PointCloud& pcd){
//     return py::array_t<float>(
//         {pcd.coords.size(), 3}, // shape
//         {sizeof(float)*3, sizeof(float)}, // strides
//         pcd.coords.data(), // pointer to data
//         py::cast(pcd)

//     );
// }   

void update_reggrow_params_with_kwargs(RegionGrowingParams& params, py::kwargs kwargs){
    // check for parameter overrides
    if (kwargs.contains("epsilon")) params.epsilon = kwargs["epsilon"].cast<float>();
    if (kwargs.contains("refit_multiplier")) params.refit_multiplier = kwargs["refit_multiplier"].cast<float>();
    if (kwargs.contains("epsilon_multiplier")) params.epsilon_multiplier = kwargs["epsilon_multiplier"].cast<float>();
    if (kwargs.contains("epsilon_multiplier_average")) params.epsilon_multiplier_average = kwargs["epsilon_multiplier_average"].cast<float>();
    if (kwargs.contains("search_radius_approx")) params.search_radius_approx = kwargs["search_radius_approx"].cast<float>();
    if (kwargs.contains("min_points_in_region")) params.min_points_in_region = kwargs["min_points_in_region"].cast<size_t>();
    if (kwargs.contains("first_refit")) params.first_refit = kwargs["first_refit"].cast<size_t>();
    if (kwargs.contains("alpha")) params.alpha = kwargs["alpha"].cast<float>();
    if (kwargs.contains("min_success_ratio")) params.min_success_ratio = kwargs["min_success_ratio"].cast<float>();
    if (kwargs.contains("max_dist_from_cent")) params.max_dist_from_cent = kwargs["max_dist_from_cent"].cast<float>();
    if (kwargs.contains("tracker_size")) params.tracker_size = kwargs["tracker_size"].cast<size_t>();
    if (kwargs.contains("oriented_normals")) params.oriented_normals = kwargs["oriented_normals"].cast<bool>();
    if (kwargs.contains("verbose")) params.verbose = kwargs["verbose"].cast<bool>();
    if (kwargs.contains("perform_cca")) params.perform_cca = kwargs["perform_cca"].cast<bool>();
}

void region_growing_chunked(
        PointCloud& pcd, 
        py::array_t<int32_t> instance_inds, 
        uint32_t chunk_size_x,
        uint32_t chunk_size_y,
        uint32_t chunk_size_z,
        py::kwargs kwargs)
{
    // initiate parameters
    std::cout << "DEBUG: init parameters" << std::endl;
    RegionGrowingParams params;  
    update_reggrow_params_with_kwargs(params, kwargs);

    // build kdtree
    std::cout << "Building KDTree" << std::endl;

    KDTreeType kdtree(3, pcd, nanoflann::KDTreeSingleIndexAdaptorParams(10/*max leaf size*/));
    kdtree.buildIndex();

    // subdivide into chunks
    std::cout << "DEBUG: subdivvide into chunks" << std::endl;
    auto chunks = subdividePointCloud(pcd, chunk_size_x, chunk_size_y, chunk_size_z);

    // initialize normals
    if (pcd.normals.empty()){
        pcd.normals = std::vector<Point>(pcd.coords.size());
    }    

    // start threads
    std::vector<std::thread> threads;
    auto buf_info = instance_inds.request();
    int32_t* instance_array = static_cast<int32_t*>(buf_info.ptr);
    std::cout << "DEBUG: creating threads loop" << std::endl;
    std::unordered_map<size_t,size_t> thread_offsets;
    for (const auto& chunk : chunks){
        std::cout << "Chunk_id: " << chunk.first << "\tNum. Points: " << chunk.second.size() << std::endl;
        threads.emplace_back(
            processChunkIndexContainer, 
            chunk.first, 
            std::ref(chunk.second), 
            std::ref(pcd), 
            std::ref(kdtree), 
            params,
            instance_array,
            std::ref(thread_offsets)
        );
    }
    
    for (auto& thread : threads){
        thread.join();
    }

    // increment each chunk to avoid overlaps
    size_t cum_sum = 0;
    for (const auto& chunk : chunks){
        for (const auto& pt_idx : chunk.second){
            instance_array[pt_idx] += cum_sum;
        }
        cum_sum += thread_offsets[chunk.first];
    }

}

PYBIND11_MODULE(reggrow, m){
    m.doc() = "Region growing module";
    py::class_<PointCloud>(m, "PointCloud")
    // PointCloud methods
        // create from ply file
        .def_static("from_ply", &PointCloud::from_ply)
        // default constructure, copies data
        .def(py::init([](py::array_t<float, py::array::c_style> coords,
                        //  py::array_t<uint8_t, py::array::c_style> colors,
                         py::object colors_optional
                        ) {
            PointCloud pcd;
            auto buf_coords = coords.request();

            //handle optional features
            if (colors_optional.is_none()){}
            else {
                auto colors = colors_optional.cast<py::array_t<uint8_t, py::array::c_style>>();
                auto buf_colors = colors.request();
                if(buf_colors.ndim != 2 || buf_colors.shape[1] != 3){
                    throw std::runtime_error("colors must be an (N,3) array");
                }
                auto begin = reinterpret_cast<Color*>(buf_colors.ptr);
                auto end = begin + buf_colors.shape[0];
                pcd.colors.assign(begin, end);
            }
            if(buf_coords.ndim != 2 || buf_coords.shape[1] != 3){
                throw std::runtime_error("coords must be an (N,3) array");
            }
            // copy the mandatory data (coords)
            {
                auto begin = reinterpret_cast<Point*>(buf_coords.ptr);
                auto end = begin + buf_coords.shape[0];
                pcd.coords.assign(begin, end);
            }



            return pcd;
        }))
        .def_property_readonly("size", &PointCloud::get_size);
        // TODO: add a copy-free constructor
        // ...
    m.def("region_growing", &region_growing_chunked, "perform region growing on a given point cloud");
        
    

        // // parameters
        // py::class_<RegionGrowingParams>(m, "RegionGrowingParams")
        //     .def(py::init([](py::kwargs kwargs) {
        //         RegionGrowingParams params;  // uses the built-in defaults
        //         if (kwargs.contains("epsilon"))
        //             params.epsilon = kwargs["epsilon"].cast<float>();
        //         if (kwargs.contains("refit_multiplier"))
        //             params.refit_multiplier = kwargs["refit_multiplier"].cast<float>();
        //         if (kwargs.contains("epsilon_multiplier"))
        //             params.epsilon_multiplier = kwargs["epsilon_multiplier"].cast<float>();
        //         if (kwargs.contains("epsilon_multiplier_average"))
        //             params.epsilon_multiplier_average = kwargs["epsilon_multiplier_average"].cast<float>();
        //         if (kwargs.contains("search_radius_approx"))
        //             params.search_radius_approx = kwargs["search_radius_approx"].cast<float>();
        //         if (kwargs.contains("min_points_in_region"))
        //             params.min_points_in_region = kwargs["min_points_in_region"].cast<size_t>();
        //         if (kwargs.contains("first_refit"))
        //             params.first_refit = kwargs["first_refit"].cast<size_t>();
        //         if (kwargs.contains("alpha"))
        //             params.alpha = kwargs["alpha"].cast<float>();
        //         if (kwargs.contains("min_success_ratio"))
        //             params.min_success_ratio = kwargs["min_success_ratio"].cast<float>();
        //         if (kwargs.contains("max_dist_from_cent"))
        //             params.max_dist_from_cent = kwargs["max_dist_from_cent"].cast<float>();
        //         if (kwargs.contains("tracker_size"))
        //             params.tracker_size = kwargs["tracker_size"].cast<size_t>();
        //         if (kwargs.contains("oriented_normals"))
        //             params.oriented_normals = kwargs["oriented_normals"].cast<bool>();
        //         if (kwargs.contains("verbose"))
        //             params.verbose = kwargs["verbose"].cast<bool>();
        //         if (kwargs.contains("perform_cca"))
        //             params.perform_cca = kwargs["perform_cca"].cast<bool>();
        //         return params;
        //     }), "Construct RegionGrowingParams from keyword arguments")
        //     .def_readwrite("epsilon", &RegionGrowingParams::epsilon)
        //     .def_readwrite("refit_multiplier", &RegionGrowingParams::refit_multiplier)
        //     .def_readwrite("epsilon_multiplier", &RegionGrowingParams::epsilon_multiplier)
        //     .def_readwrite("epsilon_multiplier_average", &RegionGrowingParams::epsilon_multiplier_average)
        //     .def_readwrite("search_radius_approx", &RegionGrowingParams::search_radius_approx)
        //     .def_readwrite("min_points_in_region", &RegionGrowingParams::min_points_in_region)
        //     .def_readwrite("first_refit", &RegionGrowingParams::first_refit)
        //     .def_readwrite("alpha", &RegionGrowingParams::alpha)
        //     .def_readwrite("min_success_ratio", &RegionGrowingParams::min_success_ratio)
        //     .def_readwrite("max_dist_from_cent", &RegionGrowingParams::max_dist_from_cent)
        //     .def_readwrite("tracker_size", &RegionGrowingParams::tracker_size)
        //     .def_readwrite("oriented_normals", &RegionGrowingParams::oriented_normals)
        //     .def_readwrite("verbose", &RegionGrowingParams::verbose)
        //     .def_readwrite("perform_cca", &RegionGrowingParams::perform_cca);
        


        
}