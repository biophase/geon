#pragma once
#include "types.h"

void read_ply(
    const std::string& file_path,
    std::vector<Point>& coords_vec, 
    std::vector<Color>* colors_vec, 
    std::vector<Point>* normals_vec, 
    bool verbose);
    
void save_ply(const std::string& file_path, const PointCloud& pcd, bool verbose);