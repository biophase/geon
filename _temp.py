from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from glob import glob
import random

import os.path as osp
import pandas as pd
from plyfile import PlyData, PlyElement
import numpy as np
from numpy.typing import NDArray

from tqdm import tqdm
import pickle

data_dir = f"./data/FWF_flat/10.0X10.0"
osp.exists(data_dir)
# generate train_test split
all_names = list(glob(osp.join(data_dir,'ply','*.ply')))
train_ratio = 0.95

N = len(all_names)
k = int(train_ratio * N)
train_i = random.sample(range(N), k) 

train_mask = np.zeros(len(all_names)).astype(bool)
train_mask[train_i] = True


all_names = np.array(all_names)
train_names = all_names[train_mask].tolist()
test_names = all_names[~train_mask].tolist()


def forwardstar_to_idx(first_edg:NDArray, adj_verts:NDArray)->NDArray:
    sources = []
    targets = []

    V = len(first_edg) - 1
    for u in range(V):
        start, end = first_edg[u], first_edg[u+1]
        src = np.full(end - start, u)
        tgt = adj_verts[start:end]
        sources.append(src)
        targets.append(tgt)
    sources = np.concat(sources)
    targets = np.concat(targets)

    return np.concat([sources[None,:], targets[None,:]],axis=0)

from typing import List, Tuple
from torch_scatter import scatter_add

def make_directed(edge_index: torch.Tensor) -> torch.Tensor:
    rev = edge_index[[1, 0], :]
    return torch.cat([edge_index, rev], dim=1)

def labels_pt2reg(y:torch.Tensor, assignment:torch.Tensor, num_classes:int,dim_size:int|None=None)->torch.Tensor:
    assert torch.all(y>=0)
    y_oh = torch.eye(num_classes)
    y_sum = scatter_add(y_oh[y], assignment, dim=0,dim_size=dim_size)
    y_reg = torch.argmax(y_sum,dim=1)
    return y_reg

class ChunkedDataset(Dataset):
    def __init__(self,
                 data_dir:str,
                 names:List[str],
                 num_classes:int,
                 ):
        
        self.data_dir = data_dir
        self.data = list()
        self.feat_names = list()
        self.names = names
        self.num_classes = num_classes

        for pcd_fp in tqdm(glob(osp.join(data_dir,'ply','*.ply'))):
            fn = osp.basename(pcd_fp).replace('.ply','')
            if fn not in self.names:
                continue
            graph_fp = osp.join(osp.split(pcd_fp)[0],'..','graph',f"{fn}.pkl")

            # load data
            pcd = pd.DataFrame(PlyData.read(pcd_fp).elements[0].data)
            pos = pcd[['x','y','z']].to_numpy(dtype=np.float32)
            feat_names = [n for n in pcd.columns if n not in ['x','y','z','labels']]
            if not len(self.feat_names): self.feat_names = feat_names
            feats = pcd[feat_names].to_numpy(dtype=np.float32)
            labels = pcd['labels'].to_numpy(dtype=np.int32)

            
            with open(graph_fp,'rb') as f:
                graph = pickle.load(f)

            
            point_feats = graph['point_feats']

            # edges
            efs = graph['edges_forwardstar']                       
            edge_index = forwardstar_to_idx(efs[0],efs[1])
            edge_index = torch.as_tensor(edge_index,dtype=torch.int64)
            edge_index = make_directed(edge_index)

            # majority vote label for each superpoint
            labels = torch.as_tensor(labels,dtype=torch.int64)
            superpoint_idx = torch.as_tensor(graph['superpoint_idx'],dtype=torch.int64)
            labels = labels_pt2reg(labels,assignment=superpoint_idx, num_classes=self.num_classes)

            self.data.append(dict(
                fn = fn,
                pos = torch.as_tensor(pos),
                feats = torch.as_tensor(feats),
                labels = labels,
                superpoint_idx = superpoint_idx,
                edge_index = edge_index,
                point_feats = torch.as_tensor(point_feats),
            ))

    def __getitem__(self,idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)

        

            


train_ds = ChunkedDataset(data_dir,[osp.basename(n).replace('.ply','') for n in train_names],num_classes=89) # type: ignore
test_ds = ChunkedDataset(data_dir,[osp.basename(n).replace('.ply','') for n in test_names],num_classes=89) # type: ignore

class PointNetEncoder(nn.Module):
    def __init__(self,
                 in_dim:int,
                 latent_dim:int,
                 hidden: int = 64,
                 dropout: float=0.1
                 ):
        super().__init__()
        
        self.lin1   = nn.Linear(in_dim, hidden)
        self.bn1    = nn.LayerNorm(hidden)
        

        self.lin2   = nn.Linear(hidden, latent_dim)
        self.bn2    = nn.LayerNorm(latent_dim)

        self.drop   = nn.Dropout(dropout)

    def forward(self, 
                pos: torch.Tensor,
                point_feats: torch.Tensor,
                super_idx: torch.Tensor                
                ):

        # prepare

        x = torch.cat([pos, point_feats], dim=-1)
        
        # layer 1
        x = self.drop(F.relu(self.bn1(self.lin1(x))))
        # layer 2
        x = self.drop(F.relu(self.bn2(self.lin2(x))))

        # pool
        super_embedding = global_max_pool(x, super_idx)

        return super_embedding
    
class SPConv(nn.Module):
    def __init__(self,
                 d:int,
                 d_e:int):
        super().__init__()
        
        # edge encoder
        self.phi_e = nn.Sequential(
            nn.Linear(3, d_e), nn.ReLU(),
            nn.Linear(d_e, d_e), nn.ReLU()
        )

        # message MLP
        self.phi_m = nn.Sequential(
            nn.Linear(2*d + d_e, d), nn.ReLU(),
            nn.Linear(d,d)
        )

        # update MLP
        self.phi_u = nn.Sequential(
            nn.Linear(2*d,d) , nn.ReLU()
        )

    def forward(self, 
                z,          # [R,d]
                centroids,  # [R,3]
                edge_index  # [2,E]
                ):
        src, dst = edge_index 

        # edge feats
        delta = centroids[dst] - centroids[src] # [E,3]
        e = self.phi_e(delta)                   # [E, d_3]

        # messages
        m = self.phi_m(torch.cat([z[src], z[dst], e], dim=-1))  # [E,d]

        # aggregation
        M = scatter_mean(m, dst, dim=0, dim_size=z.size(0))     # [R,d]

        # update
        out = self.phi_u(torch.cat([z,M], dim=-1))  # [R,d]
        return out





class SimpleModel(nn.Module):
    def __init__(self,
                 in_dim:int,
                 num_classes:int,

                 pn_latent_dim:int = 128,
                 pn_hidden: int = 128,
                 pn_dropout: float=0.1,

                 super_hidden:int = 128,
                 super_latent_dim:int = 32,
                 super_dropout: float = 0.1,

                 ):
        
        super().__init__()
        self.encoder = PointNetEncoder(in_dim,pn_latent_dim,pn_hidden,pn_dropout)

        self.lin1   = nn.Linear(pn_latent_dim,super_hidden)
        self.bn1    = nn.LayerNorm(super_hidden)

        self.lin2   = nn.Linear(super_hidden, super_latent_dim)
        self.bn2    = nn.LayerNorm(super_latent_dim)

        self.dropout = nn.Dropout(super_dropout)

        self.classifier = nn.Linear(super_latent_dim, num_classes)

    def forward(self,
                pos: torch.Tensor,
                point_feats: torch.Tensor,
                super_idx: torch.Tensor                
                ):
        # prepare inputs
        super_centroids = scatter_mean(pos, super_idx, dim=0)
        pos_local = pos - super_centroids[super_idx]

        # run model
        x = self.encoder(pos_local, point_feats, super_idx)

        x = self.dropout(F.relu(self.bn1(self.lin1(x))))
        z = self.dropout(F.relu(self.bn2(self.lin2(x))))

        return self.classifier(z), z


class SPConvModel(nn.Module):
    def __init__(
            self,
            in_dim:int,
            num_classes:int,

            pn_hidden: int = 128,
            pn_dropout: float=0.1,

            super_hidden:int=128,
            super_dropout:float=0.2,
                
            edge_dim:int = 64,
            super_num_conv:int=3,
                 ):
        super().__init__()
        self.encoder = PointNetEncoder(in_dim,super_hidden,pn_hidden,pn_dropout)

        self.conv_blocks = nn.ModuleList([
            SPConv(super_hidden, edge_dim) for _ in range(super_num_conv)
        ])
        self.ln_blocks = nn.ModuleList([
            nn.LayerNorm(super_hidden) for _ in range(super_num_conv)
        ])
        
        self.classifier = nn.Linear(super_hidden,num_classes)
        self.dropout = nn.Dropout(super_dropout)

    def forward(self,
                pos: torch.Tensor,
                point_feats: torch.Tensor,
                super_idx: torch.Tensor,
                edge_idx: torch.Tensor,                
                ):
        # prepare inputs
        super_centroids = scatter_mean(pos, super_idx, dim=0)
        pos_local = pos - super_centroids[super_idx]

        # run model
        x = self.encoder(pos_local, point_feats, super_idx)
        z = x
        for conv, ln in zip(self.conv_blocks, self.ln_blocks):
            x = conv(x, super_centroids, edge_idx)
            x = ln(x + z)
            x = F.relu(x)
            x = self.dropout(x)
            z = x

        x = self.classifier(x)

        return x, z



config_max_epochs = 50
config_device = 'cuda'
config_margin = 1.0
config_lambda_c = 0.1

train_dl = DataLoader(train_ds,batch_size=1,shuffle=True, collate_fn=lambda x: x[0])
test_dl = DataLoader(test_ds,batch_size=1, collate_fn=lambda x: x[0])

model = SPConvModel(
    in_dim = 3+12,
    num_classes=89,
    pn_hidden=64,
    super_hidden=64,
    pn_dropout=0.1,
    super_dropout=0.2,
    edge_dim=32,
    super_num_conv=3
).to(device=config_device)

optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(1,config_max_epochs+1):
    model.train()
    total_loss = 0.
    for d in tqdm(train_dl,total=len(train_ds), desc=f"Epoch {epoch:03d}"):
        pos         = d['pos'].to(device=config_device)
        point_feats = d['point_feats'].to(device=config_device)
        labels      = d['labels'].to(device=config_device)
        super_idx   = d['superpoint_idx'].to(device=config_device)
        edge_idx    = d['edge_index'].to(device=config_device)

        S = super_idx.amax() + 1 
        logits, z = model(pos, point_feats, super_idx, edge_idx)
        L_class = F.cross_entropy(logits, labels)

        # contrastive loss
        src, dst = edge_idx
        zi, zj = z[src], z[dst]
        dist2 = (zi - zj).pow(2).sum(dim=1)
        same = (labels[src] == labels[dst]).float() # [E]
        # pull
        loss_pos = same * dist2
        # push
        loss_neg = (1-same) * F.relu(config_margin - dist2)

        L_contrast = (loss_pos + loss_neg).mean()

        # total loss
        loss = L_class + config_lambda_c * L_contrast
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:02d}  average loss: {total_loss/len(train_dl):.4f}")







    