#include "split_between.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include "Eigen/Dense"

namespace {

struct Plane {
    Point centroid = Point::Zero();
    Point normal = Point::Zero();
};

Plane fit_plane(const PointCloud& pcd, const std::vector<size_t>& indices){
    if (indices.size() < 3){
        throw std::runtime_error("Plane-distance sets must contain at least three points");
    }
    Plane plane;
    for (const size_t idx : indices){
        plane.centroid += pcd.coords[idx];
    }
    plane.centroid /= static_cast<float>(indices.size());
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const size_t idx : indices){
        const Point centered = pcd.coords[idx] - plane.centroid;
        covariance += centered * centered.transpose();
    }
    covariance /= static_cast<float>(indices.size());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success || solver.eigenvalues()[1] <= 1e-12f){
        throw std::runtime_error("Plane-distance sets must contain non-collinear points");
    }
    plane.normal = solver.eigenvectors().col(0).normalized();
    if (!plane.normal.allFinite()){
        throw std::runtime_error("Could not fit a valid plane");
    }
    return plane;
}

PointCloud subset_cloud(const PointCloud& pcd, const std::vector<size_t>& indices){
    PointCloud result;
    result.coords.reserve(indices.size());
    for (const size_t idx : indices){
        result.coords.push_back(pcd.coords[idx]);
    }
    return result;
}

} // namespace

void SplitBetweenProgressState::reset(int64_t total_count){
    done.store(0, std::memory_order_relaxed);
    total.store(total_count, std::memory_order_relaxed);
    cancel.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex);
    stage = "Preparing...";
}
void SplitBetweenProgressState::requestCancel(){ cancel.store(true, std::memory_order_relaxed); }
bool SplitBetweenProgressState::isCancelled() const{ return cancel.load(std::memory_order_relaxed); }
int64_t SplitBetweenProgressState::completed() const{ return done.load(std::memory_order_relaxed); }
int64_t SplitBetweenProgressState::totalCount() const{ return total.load(std::memory_order_relaxed); }
void SplitBetweenProgressState::setStage(const std::string& value, int64_t done_count){
    done.store(done_count, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex);
    stage = value;
}
std::string SplitBetweenProgressState::stageText() const{
    std::lock_guard<std::mutex> lock(mutex);
    return stage;
}

SplitBetweenResult split_between_impl(
    const PointCloud& pcd,
    const std::vector<size_t>& working_indices,
    const std::vector<size_t>& a_indices,
    const std::vector<size_t>& b_indices,
    SplitBetweenCondition condition,
    SplitBetweenProgressState* progress
){
    const auto start = std::chrono::steady_clock::now();
    if (working_indices.empty() || a_indices.empty() || b_indices.empty()){
        throw std::runtime_error("Working, A, and B sets must all be nonempty");
    }
    SplitBetweenResult result;
    result.assignment.resize(working_indices.size(), 0);
    if (progress != nullptr){
        progress->reset(static_cast<int64_t>(working_indices.size()));
    }

    Plane plane_a;
    Plane plane_b;
    PointCloud cloud_a;
    PointCloud cloud_b;
    std::unique_ptr<SplitBetweenKDTree> tree_a;
    std::unique_ptr<SplitBetweenKDTree> tree_b;
    if (condition == SplitBetweenCondition::PlaneDistance){
        if (progress != nullptr){ progress->setStage("Fitting planes...", 0); }
        plane_a = fit_plane(pcd, a_indices);
        plane_b = fit_plane(pcd, b_indices);
    } else {
        if (progress != nullptr){ progress->setStage("Building neighbor indices...", 0); }
        cloud_a = subset_cloud(pcd, a_indices);
        cloud_b = subset_cloud(pcd, b_indices);
        tree_a = std::make_unique<SplitBetweenKDTree>(
            3, cloud_a, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree_b = std::make_unique<SplitBetweenKDTree>(
            3, cloud_b, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree_a->buildIndex();
        tree_b->buildIndex();
    }
    if (progress != nullptr){ progress->setStage("Assigning working points...", 0); }

    for (size_t i = 0; i < working_indices.size(); ++i){
        if (progress != nullptr && i % 10000 == 0 && progress->isCancelled()){
            result.cancelled = true;
            return result;
        }
        const Point& point = pcd.coords[working_indices[i]];
        float distance_a = 0.0f;
        float distance_b = 0.0f;
        if (condition == SplitBetweenCondition::PlaneDistance){
            distance_a = std::abs((point - plane_a.centroid).dot(plane_a.normal));
            distance_b = std::abs((point - plane_b.centroid).dot(plane_b.normal));
        } else {
            uint32_t index = 0;
            tree_a->knnSearch(point.data(), 1, &index, &distance_a);
            tree_b->knnSearch(point.data(), 1, &index, &distance_b);
        }
        if (distance_b < distance_a){
            result.assignment[i] = 1;
            ++result.assigned_to_b;
        } else {
            ++result.assigned_to_a;
        }
        if (progress != nullptr && i % 1000 == 0){
            progress->done.store(static_cast<int64_t>(i + 1), std::memory_order_relaxed);
        }
    }
    if (progress != nullptr){
        progress->setStage("Done", static_cast<int64_t>(working_indices.size()));
    }
    result.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}
