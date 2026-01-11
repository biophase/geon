#include <chrono>
#include <iostream>

#include "types.h"
#include "rgrow.h"
#include "io.h"
#include "connected_components.h"



size_t unionFindFind(size_t x, std::unordered_map<size_t, size_t>& parent){

    if(parent[x] != x){
        parent[x] = unionFindFind(parent[x], parent);
    }
    return parent[x];
}


void unionFindUnion (size_t x, size_t y,
    std::unordered_map<size_t, size_t>& parent,
    std::unordered_map<size_t, size_t>& ranks){
    
    // find roots
    x = unionFindFind(x, parent);
    y = unionFindFind(y, parent);
    
    if (x == y) return;

    // make sure that 'x' is the node with the higher rank
    if (ranks[x] < ranks[y]){
        size_t _temp = x;
        x = y;
        y = _temp;
    }
    
    // update the root of the set with the smaller rank
    parent[y] = x;
    if (ranks[x] == ranks[y]){
        ranks[x] ++;
    }

}


std::vector<size_t> getChunkNeighborsFull (size_t query_hash, size_t X, size_t Y, size_t Z){
    
    auto c = unflattenIndex(query_hash, X, Y, Z);
    std::vector<size_t> neighbor_hashes (26);
    size_t neighb_idx = 0;
    std::vector<int> offsets = {-1, 0, 1};
    for (int dx : offsets){
        for (int dy : offsets){
            for (int dz : offsets){
                if (!((dx==0)&&(dy==0)&&(dz==0))){
                    neighbor_hashes[neighb_idx] = flattenIndex(c[0]+dx, X, c[1]+dy, Y, c[2]+dz, Z);
                    neighb_idx ++;
                }
            }
        }
    }
    return neighbor_hashes;
}


std::vector<std::unordered_set<size_t>> unionFindCCA (PointCloud& pcd, std::unordered_set<size_t> task_inds, float epsilon){
    
    float edge_length = epsilon *1.1547; // given a cube edge length inscribed in a sphere with radius epsilon
    auto subdiv = subdividePointCloudFixedChunkSize(pcd, task_inds, edge_length, edge_length, edge_length);

    // define union-find data structure
    std::unordered_map<size_t, size_t> parents; // chunk_id --> node parent
    std::unordered_map<size_t, size_t> ranks; // chunk_id --> tree rank

    // init data structure
    // initiate all elements as their own parents --> singelton sets
    for (auto chunk : subdiv.chunks) parents.insert(std::pair(chunk.first, chunk.first));
    // initiate all ranks to be 0
    for (auto chunk : parents) ranks.insert(std::pair(chunk.first, 0));

    // iterrate over occupied chunks and add create an union with all neighbors
    for (auto& chunk : parents){
        auto neighbors = getChunkNeighborsFull(chunk.first, subdiv.X, subdiv.Y, subdiv.Z);
        for (size_t neighbor : neighbors){
            if(parents.find(neighbor) != parents.end()){
            unionFindUnion(chunk.first, neighbor, parents, ranks);
            }
        }
    }

    std::unordered_map<size_t, std::unordered_set<size_t>> root_to_reg_idxmap; 
    
    for (const auto& [chunk_index, _] : parents){
        size_t chunk_root = unionFindFind(chunk_index, parents);
        root_to_reg_idxmap.emplace(std::pair<size_t, std::unordered_set<size_t>>(chunk_root, subdiv.chunks[chunk_index]))
            .first->second.insert(subdiv.chunks[chunk_index].begin(), subdiv.chunks[chunk_index].end());
    }

    std::vector<std::unordered_set<size_t>> inds_connected_components;

    for (const auto& [root_idx, cc_inds]: root_to_reg_idxmap){
        inds_connected_components.push_back(cc_inds);
    }
    return inds_connected_components;
}