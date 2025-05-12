from segmentation.reggrow_params import *
from segmentation.reggrow.build.lib import reggrow as rg
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
import pyvista as pv
import sys
import os.path as osp
from matplotlib import pyplot as plt
sys.path.append("./segmentation/cutpursuit/pcd-prox-split/grid-graph/python")
from grid_graph import edge_list_to_forward_star # type: ignore
from segmentation.cutpursuit.python.wrappers.cp_d0_dist import cp_d0_dist



pcd_fp = "./data/flatLabels/flatLabels_viadukt.ply"
# pcd_fp = "./data/flatLabels/flatLabels_twingen.ply"
# pcd_fp = "./data/flatLabels/flatLabels_westbahnhof.ply"
# pcd_fp = "./data/flatLabels/flatLabels_niebelungen.ply"

pcd = pd.DataFrame(PlyData.read(pcd_fp).elements[0].data)

# load and remap fields

pos = pcd[['x','y','z']].to_numpy().astype(np.float32)
rgb = pcd[['red','green','blue']].to_numpy().astype(np.float32) if 'red' in pcd.columns else np.zeros((pos.shape[0],3), np.float32)
intensity = pcd['intensity'].to_numpy().astype(np.float32) if 'intensity' in pcd.columns else np.zeros(pos.shape[0], np.float32)
label = pcd['labels'].to_numpy().astype(int) if 'labels' in pcd.columns else pcd['scalar_material'].to_numpy().astype(int)

fn = osp.basename(pcd_fp)
# remap labels
# label_bin = label_map[fn][:,1][label]



rg_pcd = rg.PointCloud(pos, None)
rg_params = ReggrowParams()
graph = rg.region_graph(rg_pcd,1,1,1, **rg_params.build_params.to_dict())
print('done')