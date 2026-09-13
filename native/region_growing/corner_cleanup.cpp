#include "corner_cleanup.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "Eigen/Dense"

namespace {

struct RegionGeometry {
    int32_t label = -1;
    std::vector<size_t> members;
    Point centroid = Point::Zero();
    Point normal = Point{0.0f, 0.0f, 1.0f};
    float slenderness = 1.0f;
    float rms_width = 0.0f;
    bool valid = false;
};

std::vector<RegionGeometry> compute_region_geometry(
    const PointCloud& pcd,
    const std::vector<int32_t>& labels,
    std::unordered_map<int32_t, size_t>& label_to_region
){
    std::vector<int32_t> unique_labels;
    for (const int32_t label : labels){
        if (label >= 0 && label_to_region.emplace(label, 0).second){
            unique_labels.push_back(label);
        }
    }
    std::sort(unique_labels.begin(), unique_labels.end());
    label_to_region.clear();

    std::vector<RegionGeometry> regions(unique_labels.size());
    for (size_t i = 0; i < unique_labels.size(); ++i){
        regions[i].label = unique_labels[i];
        label_to_region[unique_labels[i]] = i;
    }
    for (size_t point_idx = 0; point_idx < labels.size(); ++point_idx){
        const int32_t label = labels[point_idx];
        if (label >= 0){
            regions[label_to_region.at(label)].members.push_back(point_idx);
        }
    }

    for (RegionGeometry& region : regions){
        if (region.members.empty()){
            continue;
        }
        for (const size_t point_idx : region.members){
            region.centroid += pcd.coords[point_idx];
        }
        region.centroid /= static_cast<float>(region.members.size());

        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const size_t point_idx : region.members){
            const Point centered = pcd.coords[point_idx] - region.centroid;
            covariance += centered * centered.transpose();
        }
        covariance /= static_cast<float>(region.members.size());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        if (solver.info() != Eigen::Success){
            continue;
        }
        const Point eigenvalues = solver.eigenvalues();
        const float largest = eigenvalues[2];
        if (!std::isfinite(largest) || largest <= 1e-12f){
            continue;
        }
        region.normal = solver.eigenvectors().col(0).normalized();
        region.slenderness = std::sqrt(std::max(0.0f, eigenvalues[1]) / largest);
        region.rms_width = std::sqrt(std::max(0.0f, eigenvalues[1]));
        region.valid = region.normal.allFinite()
            && std::isfinite(region.slenderness)
            && std::isfinite(region.rms_width);
    }
    return regions;
}

} // namespace

void CornerCleanupProgressState::reset(int64_t total_count){
    done.store(0, std::memory_order_relaxed);
    total.store(total_count, std::memory_order_relaxed);
    cancel.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex);
    stage = "Idle";
}

void CornerCleanupProgressState::requestCancel(){ cancel.store(true, std::memory_order_relaxed); }
bool CornerCleanupProgressState::isCancelled() const{ return cancel.load(std::memory_order_relaxed); }
int64_t CornerCleanupProgressState::completed() const{ return done.load(std::memory_order_relaxed); }
int64_t CornerCleanupProgressState::totalCount() const{ return total.load(std::memory_order_relaxed); }

void CornerCleanupProgressState::setStage(const std::string& value, int64_t done_count){
    done.store(done_count, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex);
    stage = value;
}

std::string CornerCleanupProgressState::stageText() const{
    std::lock_guard<std::mutex> lock(mutex);
    return stage;
}

CornerCleanupResult cleanup_corner_regions_impl(
    const PointCloud& pcd,
    const std::vector<int32_t>& labels,
    const CornerCleanupParams& params,
    CornerCleanupProgressState* progress
){
    const auto start = std::chrono::steady_clock::now();
    CornerCleanupResult result;
    result.labels = labels;
    result.stats.num_points = static_cast<int32_t>(pcd.coords.size());
    result.stats.neighbor_radius = params.epsilon * params.neighbor_radius_factor;

    if (labels.size() != pcd.coords.size()){
        throw std::runtime_error("labels length does not match point count");
    }
    if (progress != nullptr){
        progress->reset(3);
        progress->setStage("Computing region geometry...", 0);
    }

    std::unordered_map<int32_t, size_t> label_to_region;
    std::vector<RegionGeometry> regions = compute_region_geometry(pcd, labels, label_to_region);
    result.stats.num_input_regions = static_cast<int32_t>(regions.size());

    std::vector<size_t> corner_candidates;
    std::unordered_set<int32_t> destination_labels;
    for (size_t region_idx = 0; region_idx < regions.size(); ++region_idx){
        const RegionGeometry& region = regions[region_idx];
        if (!region.valid){
            continue;
        }
        if (static_cast<int32_t>(region.members.size()) >= params.min_corner_size){
            if (region.slenderness > params.slenderness_threshold){
                ++result.stats.num_rejected_slenderness;
            } else if (region.rms_width > params.absolute_width_factor * params.epsilon){
                ++result.stats.num_rejected_absolute_width;
            } else {
                corner_candidates.push_back(region_idx);
            }
        }
        if (static_cast<int32_t>(region.members.size()) >= params.min_planar_size){
            destination_labels.insert(region.label);
        }
    }
    result.stats.num_corner_candidates = static_cast<int32_t>(corner_candidates.size());
    result.stats.num_destination_regions = static_cast<int32_t>(destination_labels.size());

    if (progress != nullptr){
        if (progress->isCancelled()){
            result.stats.cancelled = true;
            return result;
        }
        progress->setStage("Building neighborhoods and cleaning corners...", 1);
    }

    PointCloud tree_cloud;
    tree_cloud.coords = pcd.coords;
    CornerCleanupKDTree kdtree(3, tree_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree.buildIndex();
    const float neighbor_radius_sq = result.stats.neighbor_radius * result.stats.neighbor_radius;

    for (size_t candidate_idx = 0; candidate_idx < corner_candidates.size(); ++candidate_idx){
        if (progress != nullptr && progress->isCancelled()){
            result.stats.cancelled = true;
            return result;
        }
        const RegionGeometry& corner = regions[corner_candidates[candidate_idx]];
        std::unordered_set<int32_t> eligible_destination_labels;
        for (const int32_t destination_label : destination_labels){
            if (destination_label == corner.label){
                continue;
            }
            const RegionGeometry& destination = regions[label_to_region.at(destination_label)];
            if (corner.rms_width <= params.relative_width_factor * destination.rms_width){
                eligible_destination_labels.insert(destination_label);
            }
        }
        if (eligible_destination_labels.size() < 2){
            ++result.stats.num_rejected_relative_width;
            continue;
        }
        std::unordered_map<int32_t, size_t> planar_support;
        std::unordered_map<int32_t, double> planar_neighbor_distance_sum;
        std::map<std::pair<int32_t, int32_t>, size_t> pair_support;

        for (size_t member_offset = 0; member_offset < corner.members.size(); ++member_offset){
            if (progress != nullptr
                && member_offset % 10000 == 0
                && progress->isCancelled()){
                result.stats.cancelled = true;
                return result;
            }
            const size_t point_idx = corner.members[member_offset];
            std::vector<nanoflann::ResultItem<size_t, float>> neighbors;
            nanoflann::RadiusResultSet<float, size_t> result_set(neighbor_radius_sq, neighbors);
            kdtree.findNeighbors(
                result_set,
                pcd.coords[point_idx].data(),
                nanoflann::SearchParameters()
            );
            std::unordered_map<int32_t, float> neighbor_label_distances;
            for (const auto& neighbor : neighbors){
                const int32_t neighbor_label = labels[neighbor.first];
                if (neighbor_label != corner.label
                    && eligible_destination_labels.find(neighbor_label)
                        != eligible_destination_labels.end()){
                    auto [distance_it, inserted] = neighbor_label_distances.emplace(
                        neighbor_label,
                        neighbor.second
                    );
                    if (!inserted && neighbor.second < distance_it->second){
                        distance_it->second = neighbor.second;
                    }
                }
            }
            for (const auto& [neighbor_label, distance_sq] : neighbor_label_distances){
                ++planar_support[neighbor_label];
                planar_neighbor_distance_sum[neighbor_label] += distance_sq;
            }
            std::vector<int32_t> ordered_neighbor_labels;
            ordered_neighbor_labels.reserve(neighbor_label_distances.size());
            for (const auto& [neighbor_label, distance_sq] : neighbor_label_distances){
                (void)distance_sq;
                ordered_neighbor_labels.push_back(neighbor_label);
            }
            std::sort(ordered_neighbor_labels.begin(), ordered_neighbor_labels.end());
            for (size_t first_idx = 0; first_idx < ordered_neighbor_labels.size(); ++first_idx){
                for (size_t second_idx = first_idx + 1;
                     second_idx < ordered_neighbor_labels.size();
                     ++second_idx){
                    ++pair_support[{
                        ordered_neighbor_labels[first_idx],
                        ordered_neighbor_labels[second_idx]
                    }];
                }
            }
        }

        if (pair_support.empty()){
            ++result.stats.num_rejected_planar_neighbors;
            continue;
        }

        // More than two planar labels can legitimately fall inside the radius. Choose
        // the pair seen together by the most corner points. For equal support, prefer
        // the pair with the closest neighboring points, then use label order to keep
        // the result deterministic.
        std::pair<int32_t, int32_t> best_pair{-1, -1};
        size_t best_pair_support = 0;
        size_t best_individual_support = 0;
        double best_neighbor_distance = std::numeric_limits<double>::infinity();
        for (const auto& [candidate_pair, candidate_pair_support] : pair_support){
            const size_t individual_support = planar_support.at(candidate_pair.first)
                + planar_support.at(candidate_pair.second);
            const double neighbor_distance =
                planar_neighbor_distance_sum.at(candidate_pair.first)
                    / static_cast<double>(planar_support.at(candidate_pair.first))
                + planar_neighbor_distance_sum.at(candidate_pair.second)
                    / static_cast<double>(planar_support.at(candidate_pair.second));
            const bool is_better = candidate_pair_support > best_pair_support
                || (candidate_pair_support == best_pair_support
                    && individual_support > best_individual_support)
                || (candidate_pair_support == best_pair_support
                    && individual_support == best_individual_support
                    && neighbor_distance < best_neighbor_distance);
            if (is_better){
                best_pair = candidate_pair;
                best_pair_support = candidate_pair_support;
                best_individual_support = individual_support;
                best_neighbor_distance = neighbor_distance;
            }
        }
        const float support_fraction = static_cast<float>(best_pair_support)
            / static_cast<float>(corner.members.size());
        if (support_fraction <= params.dual_support_threshold){
            ++result.stats.num_rejected_dual_support;
            continue;
        }

        const int32_t planar_a_label = best_pair.first;
        const int32_t planar_b_label = best_pair.second;
        const RegionGeometry& planar_a = regions[label_to_region.at(planar_a_label)];
        const RegionGeometry& planar_b = regions[label_to_region.at(planar_b_label)];
        const bool use_plane_distance = std::abs(planar_a.normal.dot(planar_b.normal)) < 0.98f;
        for (const size_t point_idx : corner.members){
            std::vector<nanoflann::ResultItem<size_t, float>> neighbors;
            nanoflann::RadiusResultSet<float, size_t> result_set(neighbor_radius_sq, neighbors);
            kdtree.findNeighbors(
                result_set,
                pcd.coords[point_idx].data(),
                nanoflann::SearchParameters()
            );
            float distance_a = std::numeric_limits<float>::infinity();
            float distance_b = std::numeric_limits<float>::infinity();
            for (const auto& neighbor : neighbors){
                if (labels[neighbor.first] == planar_a_label){
                    distance_a = std::min(distance_a, neighbor.second);
                } else if (labels[neighbor.first] == planar_b_label){
                    distance_b = std::min(distance_b, neighbor.second);
                }
            }
            if (use_plane_distance || !std::isfinite(distance_a)){
                const float plane_distance = std::abs(
                    (pcd.coords[point_idx] - planar_a.centroid).dot(planar_a.normal)
                );
                distance_a = plane_distance * plane_distance;
            }
            if (use_plane_distance || !std::isfinite(distance_b)){
                const float plane_distance = std::abs(
                    (pcd.coords[point_idx] - planar_b.centroid).dot(planar_b.normal)
                );
                distance_b = plane_distance * plane_distance;
            }
            result.labels[point_idx] = distance_a < distance_b
                ? planar_a_label
                : (distance_b < distance_a ? planar_b_label : std::min(planar_a_label, planar_b_label));
            ++result.stats.num_corner_points_reassigned;
        }
        ++result.stats.num_corner_regions_cleaned;
    }

    if (progress != nullptr){
        progress->setStage("Done", 3);
    }
    const auto end = std::chrono::steady_clock::now();
    result.stats.elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start
    ).count();
    return result;
}
