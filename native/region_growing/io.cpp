#include "io.h"
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"


#include <iostream>
#include <fstream>
// #include <vector>
// #include <iomanip>

void read_ply(
    const std::string& file_path,
    std::vector<Point>& coords_vec, 
    std::vector<Color>* colors_vec, 
    std::vector<Point>* normals_vec, 
    bool verbose) {
    try {
        std::cout << "Loading " << file_path << std::endl;

        // Create file stream
        std::unique_ptr<std::istream> file_stream(new std::ifstream(file_path, std::ios::binary));
        if (!file_stream || file_stream->fail()) {
            throw std::runtime_error("Failed to open file stream " + file_path);
        }

        tinyply::PlyFile plyfile;
        plyfile.parse_header(*file_stream);

        std::shared_ptr<tinyply::PlyData> coords, colors, normals;
        bool ply_has_colors = true;
        bool ply_has_normals = true;

        // Request properties
        try { coords = plyfile.request_properties_from_element("vertex", {"x", "y", "z"}); }
        catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }



        if (colors_vec){
            try { colors = plyfile.request_properties_from_element("vertex", {"red", "green", "blue"}); }
            catch (const std::exception& e) { 
                std::cerr << "tinyply exception: " << e.what() << std::endl; 
                ply_has_colors = false;
                }
        }

        if (normals_vec){
            try { normals = plyfile.request_properties_from_element("vertex", {"nx", "ny", "nz"}); }
            catch (const std::exception& e) { 
                std::cerr << "tinyply exception: " << e.what() << std::endl; 
                ply_has_normals = false;
                }
        }

        plyfile.read(*file_stream);

        
        // Print file info
        if (verbose){
            std::cout << "Read " << coords->count << " points" << std::endl;
            for (const auto& info : plyfile.get_info()){
                std::cout << "Info: " << info << std::endl;
            }
            std:: cout << "Fields: ";
            for (const auto& el : plyfile.get_elements()){
                std::cout << el.name << ": ";
                for (const auto& property : el.properties){
                    std::cout << property.name << ", ";
                }
            }
            std::cout << std::endl;
        }
        
        
        // Resize vectors & copy data
        // coords
        // dtype check
        if (coords->t != tinyply::Type::FLOAT32) {
            throw std::runtime_error("Unexpected data type for coordinates");
        }
        coords_vec.resize(coords->count);
        float* raw_coords = reinterpret_cast<float*>(coords->buffer.get());
        for (size_t i = 0; i < coords->count; ++i) {
            coords_vec[i] = Eigen::Vector3f(raw_coords[i * 3], raw_coords[i * 3 + 1], raw_coords[i * 3 + 2]);
        }


        // colors
        if (colors_vec && ply_has_colors){
            // dtype check
            if (colors->t != tinyply::Type::UINT8) {
                throw std::runtime_error("Unexpected data type for colors");
            }
            colors_vec->resize(colors->count);
            uint8_t* raw_colors = reinterpret_cast<uint8_t*>(colors->buffer.get());
            for (size_t i = 0; i < colors->count; ++i) {
                colors_vec->at(i) = Color{raw_colors[i * 3], raw_colors[i * 3 + 1], raw_colors[i * 3 + 2]};
            }
        } 

        // normlas
        if (normals_vec && ply_has_normals){
            // dtype check
            if (normals->t != tinyply::Type::FLOAT32) {
                throw std::runtime_error("Unexpected data type for normals");
            }
            normals_vec->resize(normals->count);
            float* raw_normals = reinterpret_cast<float*>(normals->buffer.get());
            for (size_t i = 0; i < normals->count; ++i) {
               normals_vec->at(i) = Eigen::Vector3f(raw_normals[i * 3], raw_normals[i * 3 + 1], raw_normals[i * 3 + 2]);
            }
        } 

        // unsafe copy
        // std::memcpy(coords_vec.data(), coords->buffer.get(), coords->buffer.size_bytes());
        // std::memcpy(colors_vec.data(), colors->buffer.get(), colors->buffer.size_bytes());


    

    } catch (const std::exception& e) {
        std::cerr << "Error loading the ply file: " << e.what() << std::endl;
    }
}

void save_ply(const std::string& file_path, const PointCloud& pcd, bool verbose) {
    try {
        std::filebuf fb_binary;
        fb_binary.open(file_path, std::ios::out | std::ios::binary);
        if (!fb_binary.is_open()) {
            throw std::runtime_error("Failed to open file stream " + file_path);
        }
        std::ostream outstream(&fb_binary);

        tinyply::PlyFile plyfile;

        // Add vertex positions (coords)
        plyfile.add_properties_to_element(
            "vertex", 
            {"x", "y", "z"}, 
            tinyply::Type::FLOAT32, 
            pcd.coords.size(), 
            reinterpret_cast<const uint8_t*>(pcd.coords.data()), 
            tinyply::Type::INVALID, 0);

        // Add vertex colors
        plyfile.add_properties_to_element(
            "vertex", 
            {"red", "green", "blue"}, 
            tinyply::Type::UINT8, 
            pcd.colors.size(), 
            reinterpret_cast<const uint8_t*>(pcd.colors.data()), 
            tinyply::Type::INVALID, 0);

        // Add vertex normals if provided
        if (!pcd.normals.empty()) {
            plyfile.add_properties_to_element(
                "vertex", 
                {"nx", "ny", "nz"}, 
                tinyply::Type::FLOAT32, 
                pcd.normals.size(), 
                reinterpret_cast<const uint8_t*>(pcd.normals.data()), 
                tinyply::Type::INVALID, 0);
        }
        if (!pcd.region_size.empty()) {
            plyfile.add_properties_to_element(
                "vertex", 
                {"region_size"}, 
                tinyply::Type::UINT32,
                pcd.region_size.size(), 
                reinterpret_cast<const uint8_t*>(pcd.region_size.data()), 
                tinyply::Type::INVALID, 0);
        }

        // Add optional comments
        plyfile.get_comments().push_back("Generated by tinyply");

        // Write the PLY file as binary
        plyfile.write(outstream, true);

        if (verbose) std::cout << "Saved PLY file to " << file_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error saving the PLY file: " << e.what() << std::endl;
    }
}