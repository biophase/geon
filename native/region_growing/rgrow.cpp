#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <chrono>
#include <thread>







#include "types.h"
#include "io.h"
#include "rgrow.h"
#include "connected_components.h"







class AttemptsTracker{
    /* Class to keep track of successfull region growing attempts */
    private:
        Eigen::VectorXi attempts; // 1 = successful; 0 = failed
        size_t max_num_elements;
        size_t current_index;

        void addElement(size_t attempt){
            this->attempts(current_index) = attempt;
            // rolling index
            current_index = (current_index + 1) % max_num_elements; 
        }
    public:
        explicit AttemptsTracker(size_t max_num_elements) : 
            attempts(Eigen::VectorXi::Ones(max_num_elements)),
            max_num_elements(max_num_elements),
            current_index(0){}

        void addFailedAttempt(){
            addElement(0);
        }
        void addSuccessfullAttempt(){
            addElement(1);
        }
        float getSuccessRatio(){
            return this->attempts.cast<float>().mean();
        }

    
};





void computeNormals(
    const std::vector<Point>& coords, 
    std::vector<Point>& normals,
    const std::unordered_set<size_t>& task_inds,
    KDTreeType& kdtree,
    float search_radius
    ){
        /* computes the normals of a point cloud */
        for(const size_t& i : task_inds){
            //  get neighbors
            IndicesDistType indices_dist;
            RadiusSetType result_set(search_radius, indices_dist);
            kdtree.findNeighbors(result_set, coords.at(i).data(),nanoflann::SearchParameters(.0f, false));
            Eigen::Matrix<float,Eigen::Dynamic,3> neighbors(indices_dist.size(), 3);
            for (size_t k=0; k<indices_dist.size(); ++k){
                neighbors.row(k) = coords[indices_dist[k].first] - coords[i];
            }
            // calculate normals
            Eigen::Matrix3f covariance = neighbors.transpose() * neighbors;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
            normals[i] = solver.eigenvectors().col(0);
    }
}


template <typename T>
T randomPop (std::unordered_set<T>& input){
    /*Removes a random element from a set and returns it*/
    if (input.empty()){
        throw std::range_error("Set is empty");
    }
    auto it = input.begin();
    std::advance(it, rand() % input.size());
    T value = *it;
    input.erase(it);
    return value;
}


std::pair<Eigen::Vector3f, Eigen::Vector3f> getPcdAabb(const PointCloud& pcd, std::unordered_set<size_t>& task_inds){
    /* return the Axis-Aligned Bounding Box (AABB) of a point cloud */
    Eigen::Vector3f min_values(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    );
        Eigen::Vector3f max_values(
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::min()
    );
    for (const size_t& ind : task_inds){
        min_values = min_values.cwiseMin(pcd.coords[ind]);
        max_values = max_values.cwiseMax(pcd.coords[ind]);
    }
    
    // pad the values with a small constant to facilitate better boundary detections
    min_values -= Eigen::Vector3f::Constant(1e-3f);
    max_values += Eigen::Vector3f::Constant(1e-3f);

    return std::pair<Eigen::Vector3f, Eigen::Vector3f>(min_values, max_values);
}

std::pair<Eigen::Vector3f, Eigen::Vector3f> getPcdAabb(const PointCloud& pcd){
    /* return the Axis-Aligned Bounding Box (AABB) of a point cloud */
    Eigen::Vector3f min_values(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    );
        Eigen::Vector3f max_values(
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::min()
    );
    for (const auto& xyz : pcd.coords){
        min_values = min_values.cwiseMin(xyz);
        max_values = max_values.cwiseMax(xyz);
    }
    
    // pad the values with a small constant to facilitate better boundary detections
    min_values -= Eigen::Vector3f::Constant(1e-3f);
    max_values += Eigen::Vector3f::Constant(1e-3f);

    return std::pair<Eigen::Vector3f, Eigen::Vector3f>(min_values, max_values);
}



std::vector<size_t> unflattenIndex(size_t flat_idx, size_t X, size_t Y, size_t Z){
    std::vector<size_t> c(3);
    // c[2] =    flat_idx % Z;
    // c[1] =  ((flat_idx - c[2]) / Z) % Y;
    // c[0] = (((flat_idx - c[2]) / Z) - c[1]) / Y;

    c[2] = flat_idx % Z;
    c[1] = (flat_idx / Z) % Y;
    c[0] = flat_idx / (Y * Z);

    return c;

}

size_t flattenIndex (size_t x, size_t X, size_t y, size_t Y, size_t z, size_t Z){
    return x*Y*Z + y*Z + z; 
}



subdividePointCloudFixedChunkSize_returnType subdividePointCloudFixedChunkSize(const PointCloud& pcd,
    std::unordered_set<size_t> task_inds, float chunk_size_x, float chunk_size_y, float chunk_size_z 
    )
    {
        const auto [aabb_min_values, aabb_max_values] = getPcdAabb(pcd, task_inds);
        std::unordered_map<size_t, std::unordered_set<size_t>> chunks;
        Eigen::Vector3f aabb_size = aabb_max_values-aabb_min_values;  

        size_t num_chunks_x = ceil(aabb_size[0] / chunk_size_x);
        size_t num_chunks_y = ceil(aabb_size[1] / chunk_size_y);
        size_t num_chunks_z = ceil(aabb_size[2] / chunk_size_z);
        
        Eigen::Vector3f chunk_size = {chunk_size_x, chunk_size_y, chunk_size_z};

        // for (size_t i=0; i < pcd.coords.size(); ++i){
        for (size_t i : task_inds){
            Eigen::Vector3i chunk_coord = (pcd.coords[i]-aabb_min_values).cwiseQuotient(chunk_size).cast<int>();
            size_t coord_hash = flattenIndex(
                chunk_coord[0], num_chunks_x,
                chunk_coord[1], num_chunks_y,
                chunk_coord[2], num_chunks_z);
            chunks.emplace(coord_hash, std::unordered_set<size_t>{}).first->second.insert(i);
        }
        return subdividePointCloudFixedChunkSize_returnType {chunks, num_chunks_x, num_chunks_y, num_chunks_z};
    }


std::unordered_map<size_t, std::unordered_set<size_t>> subdividePointCloud(const PointCloud& pcd, 
    size_t num_chunks_x, size_t num_chunks_y, size_t num_chunks_z 
    ){
        const auto [aabb_min_values, aabb_max_values] = getPcdAabb(pcd);
        
        std::unordered_map<size_t, std::unordered_set<size_t>> chunks;
        Eigen::Vector3f aabb_size = aabb_max_values-aabb_min_values;      
        Eigen::Vector3f chunk_size = {
            aabb_size[0]/num_chunks_x,
            aabb_size[1]/num_chunks_y,
            aabb_size[2]/num_chunks_z,            
        };
         
        for (size_t i=0; i < pcd.coords.size(); ++i){
            Eigen::Vector3i chunk_coord = (pcd.coords[i]-aabb_min_values).cwiseQuotient(chunk_size).cast<int>();
            size_t coord_hash = flattenIndex(
                chunk_coord[0], num_chunks_x,
                chunk_coord[1], num_chunks_y,
                chunk_coord[2], num_chunks_z);
            chunks.emplace(coord_hash, std::unordered_set<size_t>{}).first->second.insert(i);
        }
        return chunks;

}

void refitPlane(Region& reg, PointCloud& pcd){
    Eigen::Matrix<float, Eigen::Dynamic, 3> reg_coords(reg.indices.size(), 3);
    Point new_reg_centroid = {0, 0, 0};
    
    size_t idx_out = 0;
    for (const auto& ind : reg.indices){
        reg_coords.row(idx_out) = pcd.coords[ind];
        new_reg_centroid += pcd.coords[ind];
        idx_out ++;
    }
    
    new_reg_centroid /= reg.indices.size();
    reg_coords.rowwise() -= new_reg_centroid.transpose();
    Eigen::Matrix3f covariance = reg_coords.transpose() * reg_coords;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    reg.normal = solver.eigenvectors().col(0);
    reg.centroid = new_reg_centroid;
}




regionGrowing_returnType regionGrowing( 
    PointCloud& pcd,
    KDTreeType& kdtree,
    const std::unordered_set<size_t>* task_indices, // indices to segment; if nullptr -> segment all
    RegionGrowingParams params
){
    auto& p = params;
    float cos_alpha = std::cos(p.alpha);
    // init
    std::unordered_set<size_t> unassigned_inds;
    std::vector<Region> regions;
    

    // initiate task indices in pcd
    if (task_indices == nullptr) for (size_t i=0; i<pcd.coords.size(); ++i) unassigned_inds.insert(i);
    else unassigned_inds = *task_indices;


    


    auto attempts_tracker = AttemptsTracker(p.tracker_size);

    while(!unassigned_inds.empty() && attempts_tracker.getSuccessRatio() >= p.min_success_ratio){
        // init new region
        size_t seed_idx = randomPop<size_t>(unassigned_inds);
        Region reg;
        std::unordered_set<size_t> front_inds;
        std::unordered_set<size_t> cand_inds_buffer; 
        size_t next_refit = p.first_refit;
        
        
        // handle seed point
        reg.indices.insert(seed_idx);
        reg.centroid = pcd.coords[seed_idx];
        reg.normal = pcd.normals[seed_idx];
        front_inds.insert(seed_idx);
        unassigned_inds.erase(seed_idx);

        // growing loop
        while (!front_inds.empty()){
            // float total_residual = 0.f;
            
            std::unordered_set<size_t> front_inds_new;
            

            // collect new neighbors of region within epsilon and save to buffer
            cand_inds_buffer.clear();
            for (const size_t& f_i : front_inds){
                IndicesDistType indices_dist;
                RadiusSetType result_set(p.epsilon, indices_dist);
                kdtree.findNeighbors(result_set, pcd.coords[f_i].data(), 
                    nanoflann::SearchParameters(p.search_radius_approx,false));
                for (const auto& i_d : indices_dist) {
                        cand_inds_buffer.insert(i_d.first);
                }
            }

            // filter valid indices and add to new front
            for (const size_t& i : cand_inds_buffer) {
                float reg_normal_dot_fnormal = pcd.normals[i].dot(reg.normal);
                Point coord_local = pcd.coords[i] - reg.centroid;
                float dist_to_plane = coord_local.dot(reg.normal);

                if (
                    (dist_to_plane <= p.epsilon_multiplier * p.epsilon) &&
                    ((std::abs(reg_normal_dot_fnormal) > cos_alpha + (1 - cos_alpha) * (coord_local.norm() / p.max_dist_from_cent) && !p.oriented_normals) ||
                    (p.oriented_normals && reg_normal_dot_fnormal > cos_alpha + (1 - cos_alpha) * coord_local.norm() / p.max_dist_from_cent)) &&
                    (unassigned_inds.find(i) != unassigned_inds.end())
                ) {
                    front_inds_new.insert(i);
                    reg.indices.insert(i);
                    unassigned_inds.erase(i);
                }
            }
                            
                // switch to new front for next iterration
                front_inds = std::move(front_inds_new);

                // refit plane if interval is reached
                if (reg.indices.size() >= next_refit) {
                    next_refit = static_cast<size_t>(next_refit * p.refit_multiplier);

                    refitPlane(reg, pcd);
                    /* moved to refitPlane(...)
                    Eigen::Matrix<float, Eigen::Dynamic, 3> reg_coords(reg.indices.size(), 3);
                    Point new_reg_centroid = {0, 0, 0};
                    size_t idx_out = 0;

                    for (const auto& ind : reg.indices) {
                        reg_coords.row(idx_out) = pcd.coords[ind];
                        new_reg_centroid += pcd.coords[ind];
                        idx_out++; 
                    }

                    new_reg_centroid /= reg.indices.size();
                    reg_coords.rowwise() -= new_reg_centroid.transpose();
                    Eigen::Matrix3f covariance = reg_coords.transpose() * reg_coords;
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);

                    reg.normal = solver.eigenvectors().col(0);
                    reg.centroid = new_reg_centroid;
                    */

                    // re-evaluate points in region
                    for (auto it = reg.indices.begin(); it != reg.indices.end();){
                        float reg_normal_dot_pnormal = pcd.normals[*it].dot(reg.normal);
                        Point coord_local = pcd.coords[*it] - reg.centroid;
                        float dist_to_plane = coord_local.dot(reg.normal);
                        
                        // rule-check
                    if (
                        // distance check
                        !(dist_to_plane <= p.epsilon_multiplier * p.epsilon) ||                                                                           
                        // angle check oriented
                        !((std::abs(reg_normal_dot_pnormal) > cos_alpha + (1-cos_alpha)*(coord_local.norm() / p.max_dist_from_cent) && !p.oriented_normals) ||
                        // angle check nont-oriented
                          (p.oriented_normals && reg_normal_dot_pnormal > cos_alpha + (1-cos_alpha)*(coord_local.norm()/p.max_dist_from_cent)))          
                        ){
                            // remove point from region
                            unassigned_inds.insert(*it);
                            it = reg.indices.erase(it);
                            // total_residual -= dist_to_plane;
                        }
                        else{
                            std::advance(it,1);
                        }
                    }
                }
            }
        
        // region terminated
        
        // check if region is valid
        if (reg.indices.size() < p.min_points_in_region){ // invalid region
            attempts_tracker.addFailedAttempt();
            
            // return region indices to unassigned
            for (const auto& idx: reg.indices){
                unassigned_inds.insert(idx);
            }
        }
        else{ // valid region
            
            
            if (params.perform_cca){ // cca for non-connected regions (e.g. due to curbs)
                auto cca_results = unionFindCCA(pcd, reg.indices, p.epsilon);
                for (const auto& reg_inds : cca_results){
                    if (reg_inds.size() >= p.min_points_in_region){
                        attempts_tracker.addSuccessfullAttempt();
                        Region sub_reg;
                        sub_reg.indices = reg_inds;
                        refitPlane(sub_reg, pcd);
                        regions.push_back(sub_reg);
                    }
                    else{
                        for (const auto& idx : reg_inds){
                            unassigned_inds.insert(idx);
                        }
                    }
                }
            }
            else{ // cca turned off
                regions.push_back(reg);
                attempts_tracker.addSuccessfullAttempt();
            }
        }
        if (p.verbose){
            std::cout 
                << "Thread id: " << std::this_thread::get_id() 
                << "\tRegion attempt:" << regions.size() 
                << "\tRegion points:" << reg.indices.size() 
                << "\tRemaining points:" << unassigned_inds.size() 
                << "\tSuccess ratio:" << attempts_tracker.getSuccessRatio() << std::endl;          
        }
    }

    // TODO: CCA on unassigned --> convert to regions

    ///////////////////
    /* edge clean-up */
    ///////////////////

    std::cout << "Starting cleanup." << std::endl;
    
    // initiate map from pcd index to region index within thread
    std::unordered_map<size_t, size_t> pcd_to_reg_idxmap;
    for (size_t reg_i =0; reg_i < regions.size(); ++reg_i){
        for (const size_t& idx : regions[reg_i].indices){
            pcd_to_reg_idxmap[idx] = reg_i;
        }
    }
    std::unordered_set<size_t> cleaned_inds;


    for (auto idx_it = unassigned_inds.begin(); idx_it != unassigned_inds.end();){
        
        // get neighbors of unassigned point
        IndicesDistType indices_dist;
        RadiusSetType result_set(p.epsilon, indices_dist);
        kdtree.findNeighbors(result_set, pcd.coords[*idx_it].data(), 
            nanoflann::SearchParameters(p.search_radius_approx,true));
        
        // relaxed rule-check
        float min_dist_to_plane = std::numeric_limits<float>::max();
        float min_dist_criterion = std::numeric_limits<float>::max();
        size_t idx_best_region;
        for (const auto& i_d : indices_dist){
            if (pcd_to_reg_idxmap.find(i_d.first) != pcd_to_reg_idxmap.end()){ // check if neighbor is in a region
                // std:: cout << "Reg index is: " << pcd_to_reg_idxmap[i_d.first] << std::endl;
                Region& cand_region = regions[pcd_to_reg_idxmap[i_d.first]];
                float dist_to_plane = std::abs((pcd.coords[*idx_it] - cand_region.centroid).dot(cand_region.normal));
                float dist_criterion = (i_d.second / p.epsilon) + (dist_to_plane / (p.epsilon * p.epsilon_multiplier)*1.);
                if ((dist_criterion < min_dist_criterion) && (cleaned_inds.find(i_d.first) == cleaned_inds.end())) { // insertion condition
                    min_dist_criterion = dist_criterion;
                    min_dist_to_plane = dist_to_plane;
                    idx_best_region = pcd_to_reg_idxmap[i_d.first];
                }
            }
        }
        if ((min_dist_to_plane < p.epsilon * p.epsilon_multiplier) && (!indices_dist.empty())){
            regions[idx_best_region].indices.insert(*idx_it);
            pcd_to_reg_idxmap[*idx_it] = pcd_to_reg_idxmap[idx_best_region];
            cleaned_inds.insert(*idx_it);
            idx_it = unassigned_inds.erase(idx_it); // assigned point --> advance iterator                
        }
        else{
            std::advance(idx_it, 1); // could not assign point --> continue
        }
    }

    return regionGrowing_returnType {std::move(regions), pcd_to_reg_idxmap, unassigned_inds};

}




using Edge = std::pair<size_t,size_t>; 

struct EdgeHash{
    size_t operator() (const Edge& edge) const{
        size_t a = edge.first;
        size_t b = edge.second;
        if (a>b) std::swap(a,b);
        return std::hash<size_t>{}(a) ^ (std::hash<size_t>{}(b) << 1);
    }
};

struct EdgeEqual{
    bool operator()(const Edge& a, const Edge& b) const{
        return (((a.first == b.first) && (a.second == b.second)) ||
                ((a.first == b.second) && (a.second == b.first)));
    }
};

struct Graph{
    std::vector<Region> regions;
    std::unordered_map<Edge, std::unordered_set<size_t>, EdgeHash, EdgeEqual> edges;

    inline void reportStatistics(){
        std::cout << "Number of regions in graph: " << regions.size()
                  << "\tnumber of edges: " << edges.size() << std::endl;
    }
};

void buildGraphEdges(
    PointCloud& pcd, 
    Graph& graph, 
    KDTreeType& kdtree,
    const std::unordered_set<size_t>& task_inds,
    std::unordered_map<size_t, size_t>& pcd_to_reg_idxmap,
    RegionGrowingParams params   
){
    auto& p = params;
    for (const size_t& pcd_ind : task_inds){
        std::unordered_set<size_t> reg_neighb_inds;
        IndicesDistType neighbs_inds_dists;
        RadiusSetType result_set (p.epsilon/2, neighbs_inds_dists);
        kdtree.findNeighbors(result_set, pcd.coords.at(pcd_ind).data(), nanoflann::SearchParameters(p.search_radius_approx, false));
        
        // gather the indices of all neighboring regions 
        for (const auto& [neib_i, _] : neighbs_inds_dists){
            reg_neighb_inds.emplace(pcd_to_reg_idxmap[neib_i]);
        }

        // get all possible region pairs and add them as an edge in the graph
        for (const size_t& i : reg_neighb_inds){
            for (const size_t& j : reg_neighb_inds){
                // Edge, std::unordered_set<size_t>
                graph.edges.emplace(Edge(i,j), std::unordered_set<size_t>({})).first->second.emplace(pcd_ind);
            }
        }  
    }
}







void processChunk(
    size_t chunk_id,
    const std::unordered_set<size_t>& chunk_indices,
    PointCloud& pcd,
    KDTreeType& kdtree,
    RegionGrowingParams params
) {
    computeNormals(pcd.coords, pcd.normals, chunk_indices, kdtree, params.epsilon);
    auto result = regionGrowing(pcd, kdtree, &chunk_indices, params);
    
    auto& remaining = result.unassigned; // not needed
    

    // build graph
    Graph graph;
    graph.regions = std::move(result.regions);
    std::cout << "Building graph." << std::endl;
    buildGraphEdges(pcd, graph, kdtree, chunk_indices, result.pcd_to_reg_idxmap, params);
    graph.reportStatistics();
    

    // save segmented point cloud
    for (size_t reg_i = 0; reg_i < graph.regions.size(); ++reg_i) {
        PointCloud pcd_reg;
        for (const auto& idx : graph.regions[reg_i].indices) {
            pcd_reg.coords.push_back(pcd.coords[idx]);
            pcd_reg.colors.push_back(pcd.colors[idx]);
            pcd_reg.normals.push_back(pcd.normals[idx]);
        }
        save_ply("./test_outputs/segments/chunk_" + std::to_string(chunk_id) + "_seg_" + std::to_string(reg_i) + ".ply", pcd_reg, false);
    }

    // save remaining unassigned points
    PointCloud pcd_reg;
    for (const auto& idx : remaining) {
        pcd_reg.coords.push_back(pcd.coords[idx]);
        pcd_reg.colors.push_back(pcd.colors[idx]);
        pcd_reg.normals.push_back(pcd.normals[idx]);
    }
    save_ply("./test_outputs/segments/_chunk_" + std::to_string(chunk_id) + "_remaining.ply", pcd_reg, false);
}

void processChunkIndexContainer(
    size_t chunk_id,
    const std::unordered_set<size_t>& chunk_indices,
    PointCloud& pcd,
    KDTreeType& kdtree,
    RegionGrowingParams params,
    int32_t* output,
    std::unordered_map<size_t,size_t>& thread_offsets
){
    std::cout << "DEBUG: computing normals" << std::endl;
    /* Simple version of reggrow intended for annotating point clouds. Does not use the graph */
    computeNormals(pcd.coords, pcd.normals, chunk_indices, kdtree, params.epsilon);
    auto result = regionGrowing(pcd, kdtree, &chunk_indices, params);
    for (uint32_t reg_idx = 0; reg_idx < result.regions.size(); ++reg_idx){
        Region& region = result.regions[reg_idx];
        for (auto& point_idx : region.indices){
            output[point_idx] = reg_idx;
        }
    }
    // save number of regions to offset global index
    thread_offsets[chunk_id] = result.regions.size();
}