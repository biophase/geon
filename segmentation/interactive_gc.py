import sys
import numpy as np
from typing import Tuple, List, Dict, Optional
from numpy.typing import NDArray
import maxflow
from segmentation.cutpursuit.python.wrappers.cp_d0_dist import cp_d0_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_add, scatter_mean
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def graph_distance_embedding(
    coords: np.ndarray,
    edges: Tuple[np.ndarray, np.ndarray],
    dim: int = 16,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Compute diffusion-map embedding of a graph with CSR edges.
    """
    N = coords.shape[0]
    indptr, indices = edges
    if indptr.size != N+1:
        raise ValueError(f"CSR indptr size {indptr.size} does not match nodes {N}+1")
    data = np.ones(indices.shape[0], dtype=np.float32)
    A = sp.csr_matrix((data, indices, indptr), shape=(N, N))
    A = 0.5 * (A + A.T)
    deg = np.array(A.sum(axis=1)).ravel()
    inv = 1.0 / np.sqrt(deg + 1e-12)
    L = sp.eye(N) - sp.diags(inv).dot(A).dot(sp.diags(inv))
    eigvals, eigvecs = spla.eigsh(L, k=min(dim+1, N-1), which='SM', tol=tol)
    return eigvecs[:, 1:dim+1]


@torch.no_grad()
def scatter_eig(src: torch.Tensor, idx: torch.Tensor, G: Optional[int]=None, eps: float=1e-6):
    if G is None:
        G = int(idx.max().item()) + 1
    N, D = src.shape
    ones = torch.ones_like(idx, dtype=src.dtype)
    cnt = scatter_add(ones, idx, dim=0, dim_size=G)
    mu = scatter_add(src, idx, dim=0, dim_size=G) / (cnt.unsqueeze(1) + eps)
    x = src.unsqueeze(-1)
    xx = x @ x.transpose(1,2)
    flat = xx.reshape(N, D*D)
    sum_xx = scatter_add(flat, idx, dim=0, dim_size=G).reshape(G, D, D)
    cov = sum_xx/(cnt.view(G,1,1)+eps) - mu.unsqueeze(-1)@mu.unsqueeze(1)
    cov = cov + eps*torch.eye(D, device=src.device).unsqueeze(0)
    vals, vecs = torch.linalg.eigh(cov)
    return vals, vecs


def build_csr(edge_idx: torch.Tensor, num_nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sys.path.append("./segmentation/cutpursuit/pcd-prox-split/grid-graph/python/bin")
    from grid_graph import edge_list_to_forward_star
    indptr, indices, reidx = edge_list_to_forward_star(
        num_nodes, edge_idx.T.contiguous().cpu().numpy()
    )
    return indptr.astype(np.uint32), indices.astype(np.uint32), reidx


class Graph:
    """
    Hierarchical graph: builds fine and coarse cut-pursuit partitions.
    """
    def __init__(
        self,
        pos: NDArray,
        feats_point: NDArray,
        cp_args_fine: Dict = {},
        cp_args_coarse: Dict = {}
    ):
        # raw data
        self.pos = pos
        # fine graph connectivity
        pos_t = torch.from_numpy(pos).float()
        edge_idx = knn_graph(pos_t, k=cp_args_fine.get('k',10))
        edge_idx = torch.cat([edge_idx, edge_idx.flip(0)], dim=1)
        src, tgt, _ = build_csr(edge_idx, pos.shape[0])

        # fine cut-pursuit (get super_fine, graph_f)
        Df = pos.shape[1] + feats_point.shape[1]
        Xf = np.asfortranarray(np.concatenate([pos - pos.mean(0), feats_point], axis=1).T,
                                dtype=np.float64)
        ew = np.ones_like(tgt, np.float64) * cp_args_fine.get('regularization', 0.1)
        vw = np.ones(pos.shape[0], np.float64)
        cw = np.ones(Df, np.float64)
        cw[:pos.shape[1]] *= cp_args_fine.get('spatial_weight', 10.0)
        sup_f, _, graph_f = cp_d0_dist(
            Df, Xf, src, tgt,
            edge_weights=ew, vert_weights=vw, coor_weights=cw,
            min_comp_weight=cp_args_fine.get('cutoff',5),
            cp_dif_tol=cp_args_fine.get('cp_dif_tol',1e-2),
            cp_it_max=cp_args_fine.get('cp_it_max',10),
            split_damp_ratio=cp_args_fine.get('split_damp_ratio',0.7),
            verbose=False, max_num_threads=0,
            balance_parallel_split=True,
            compute_List=False, compute_Graph=True, compute_Time=False
        )
        # unpack fine graph edges
        edges_f = (graph_f[0], graph_f[1])
        self.super_fine = sup_f
        self.edges = edges_f

        # compute fine superpoint features
        Rf = int(sup_f.max()) + 1
        sup_idx = torch.tensor(sup_f, dtype=torch.long)
        centroids_f = scatter_mean(pos_t, sup_idx, dim=0, dim_size=Rf).cpu().numpy()
        vals, vecs = scatter_eig(pos_t - torch.from_numpy(centroids_f)[sup_idx], sup_idx, Rf)
        l1, l2, l3 = vals[:,0], vals[:,1], vals[:,2]
        normals = vecs[:,:,0].cpu().numpy()
        feats_f = scatter_mean(torch.from_numpy(feats_point), sup_idx, dim=0, dim_size=Rf).cpu().numpy()
        feats_sup = np.concatenate([
            feats_f,
            centroids_f,
            normals,
            np.abs(normals[:,2:3]),
            ((l3-l2)/(l3+1e-6))[:,None],
            ((l2-l1)/(l3+1e-6))[:,None],
            (l1/(l3+1e-6))[:,None]
        ], axis=1)
        feats_sup = (feats_sup - feats_sup.mean(0)) / (feats_sup.std(0)+1e-6)

        # coarse cut-pursuit on fine superpoints
        src_c, tgt_c = edges_f
        Xc = np.asfortranarray(np.concatenate([centroids_f, feats_sup], axis=1).T,
                                dtype=np.float64)
        ew_c = np.ones_like(tgt_c, np.float64) * cp_args_coarse.get('regularization',0.05)
        vw_c = np.ones(Rf, np.float64)
        cw_c = np.ones(Xc.shape[0], np.float64)
        sup_c, _, graph_c = cp_d0_dist(
            Xc.shape[0], Xc, src_c, tgt_c,
            edge_weights=ew_c, vert_weights=vw_c, coor_weights=cw_c,
            min_comp_weight=cp_args_coarse.get('cutoff',50),
            cp_dif_tol=cp_args_coarse.get('cp_dif_tol',1e-2),
            cp_it_max=cp_args_coarse.get('cp_it_max',10),
            split_damp_ratio=cp_args_coarse.get('split_damp_ratio',0.7),
            verbose=False, max_num_threads=0,
            balance_parallel_split=True,
            compute_List=False, compute_Graph=True, compute_Time=False
        )
        edges_c = (graph_c[0], graph_c[1])
        self.super_coarse = sup_c
        # compute coarse superpoint centroids
        sup_c_idx = torch.tensor(sup_c, dtype=torch.long)
        centroids_c = scatter_mean(torch.from_numpy(centroids_f), sup_c_idx, dim=0,
                                   dim_size=int(sup_c.max())+1).cpu().numpy()
        # diffusion embedding on coarse graph
        coarse_emb = graph_distance_embedding(centroids_c, edges_c, dim=8)
        coarse_emb -= coarse_emb.mean(axis=0)
        coarse_emb /= np.std(coarse_emb, axis=0)
        self.coarse_map = sup_c
        # final features for graph-cut
        # self.feats = np.concatenate([feats_sup, coarse_emb[self.coarse_map]], axis=1)
        
        self.feats = np.concatenate([feats_sup], axis=1)
        print(f"DEBUG built graph with fine feats of shape {self.feats.shape=}")


class TinyUnaryMLP(nn.Module):
    """Tiny MLP for unary potentials."""
    def __init__(self, in_dim: int, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class InteractiveGraphCut:
    def __init__(
        self,
        graph: Graph,
        smoothing_beta: float = 10,
        pairwise_lambda: float = .1
    ):
        self.graph = graph
        self.beta = smoothing_beta
        self.lmbda = pairwise_lambda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'DEBUG starting interactive cut on graph with feat shape {graph.feats.shape}')
        feats = torch.from_numpy(graph.feats).float().to(self.device)
        self.unary = TinyUnaryMLP(feats.shape[1]).to(self.device)
        self.opt = torch.optim.Adam(self.unary.parameters(), lr=1e-4)
        self.crit = nn.CrossEntropyLoss()
        self.pos: List[int] = []
        self.neg: List[int] = []

    def register_pick(self, pid: int, positive: bool) -> NDArray:
        sp = self.graph.super_fine[pid]
        (self.pos if positive else self.neg).append(sp)
        if self.pos and self.neg:
            self._train_unary()
        labels = self._solve()
        return labels[self.graph.super_fine]

    def _train_unary(self, epochs: int = 100):
        feats = torch.from_numpy(self.graph.feats).float().to(self.device)
        mask = torch.zeros(feats.size(0), dtype=torch.bool, device=self.device)
        mask[self.pos + self.neg] = True
        lbl = torch.zeros(feats.size(0), dtype=torch.long, device=self.device)
        lbl[self.pos] = 1
        lbl[self.neg] = 0
        for _ in range(epochs):
            self.opt.zero_grad()
            out = self.unary(feats[mask])
            loss = self.crit(out, lbl[mask])
            loss.backward()
            self.opt.step()

    def compute_unary(self) -> Tuple[np.ndarray,np.ndarray]:
        feats = torch.from_numpy(self.graph.feats).float().to(self.device)
        logp = F.log_softmax(self.unary(feats), dim=1).detach().cpu().numpy()
        return -logp[:,1], -logp[:,0]

    def build_graph(self, cost_source: np.ndarray, cost_sink: np.ndarray) -> maxflow.Graph:
        V = self.graph.feats.shape[0]
        fe, av = self.graph.edges
        g = maxflow.Graph[float](V, av.size)
        g.add_nodes(V)
        for u in range(V):
            g.add_tedge(u, cost_source[u], cost_sink[u])
        for u in range(V):
            for idx in range(fe[u], fe[u+1]):
                v = av[idx]
                diff = self.graph.feats[u] - self.graph.feats[v]
                w = self.lmbda * np.exp(-self.beta * (diff @ diff))
                g.add_edge(u, v, w, w)
        return g

    def _solve(self) -> np.ndarray:
        cs, ct = self.compute_unary()
        g = self.build_graph(cs, ct)
        g.maxflow()
        return np.array([g.get_segment(u) for u in range(self.graph.feats.shape[0])], dtype=bool)
