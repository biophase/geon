#pragma once
#include "Eigen/Dense"
#include <unordered_set>


constexpr double PI = 3.1415926535897932384626433832795;

using Point = Eigen::Vector3f;
struct Color{ uint8_t red, green, blue;};


struct PointCloud{
    // fields
    std::vector<Point> coords;
    std::vector<Color> colors;
    std::vector<Point> normals;
    std::vector<int32_t> region_size;

    // methods
    inline size_t kdtree_get_point_count() const {return coords.size();}
    inline float kdtree_get_pt(const size_t idx, int dim) const {return coords[idx][dim];}
    template <class BBox> bool kdtree_get_bbox(BBox& bb) const{return false;}
    static PointCloud from_ply(const std::string& file_path, bool verbose);
    size_t get_size();
}; 

struct Region{
    Point centroid;
    Point normal;
    std::unordered_set<size_t> indices;
};