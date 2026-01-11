#include "types.h"
#include "io.h"

PointCloud PointCloud::from_ply(const std::string& file_path, bool verbose)
{
        PointCloud pcd;
        read_ply(file_path, pcd.coords, &pcd.colors, nullptr /*no normals*/, verbose);
        return pcd;
    }

size_t PointCloud::get_size(){
    return this->coords.size();
}