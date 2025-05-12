import sys
import numpy as np
from typing import Tuple, List, Dict
from numpy.typing import NDArray
import maxflow
from sklearn.mixture import BayesianGaussianMixture
from segmentation.cutpursuit.python.wrappers.cp_d0_dist import cp_d0_dist
from plyfile import PlyData, PlyElement
import torch
from torch_geometric.nn import knn_graph, radius_graph
from torch_scatter import scatter_add, scatter_mean, scatter_sum
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def graph_distance_embedding(
    pos: np.ndarray,
    edges: Tuple[np.ndarray, np.ndarray],
    dim: int = 16,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Compute a spectral/diffusion‐map embedding of the graph nodes.

    Args:
        pos   : (N,3) array of node positions (used to infer N).
        edges : tuple (indptr, indices) in CSR-forward‐star form, where
                indptr has length N+1 and indices has length = #edges.
        dim   : embedding dimension (default = 16).
        tol   : tolerance for the eigensolver.

    Returns:
        emb   : (N, dim) array of embedding coordinates.
    """
    N = pos.shape[0]
    indptr, indices = edges

    # 1) Build adjacency matrix A (unweighted, undirected)
    data = np.ones(indices.shape[0], dtype=np.float32)
    A = sp.csr_matrix((data, indices, indptr), shape=(N, N))
    A = 0.5 * (A + A.T)  # ensure symmetry

    # 2) Degree matrix and normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    deg = np.array(A.sum(axis=1)).ravel()
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-12)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    I = sp.eye(N, format='csr')
    L = I - D_inv_sqrt.dot(A).dot(D_inv_sqrt)

    # 3) Compute the smallest (dim+1) eigenpairs of L
    #    'SM' → smallest magnitude. The first eigenpair is λ≈0, v=const.
    eigvals, eigvecs = spla.eigsh(L, k=dim+1, which='SM', tol=tol)

    # 4) Drop the trivial constant eigenvector (index 0) and return the next `dim`
    emb = eigvecs[:, 1:dim+1]
    return emb

@torch.no_grad()
def scatter_eigendecomposition(src: torch.Tensor, index: torch.Tensor, G: int|None = None, eps: float = 1e-6):
    """Compute per-point eigenvalues and eigenvectors of local covariance."""
    if G is None:
        G = int(index.max().item()) + 1
    N, D = src.shape
    # accumulate counts
    ones = torch.ones_like(index, dtype=src.dtype)
    counts = scatter_add(ones, index, dim=0, dim_size=G)  # [G]
    # mean
    sum_src = scatter_add(src, index, dim=0, dim_size=G)   # [G, D]
    mu = sum_src / (counts[:, None] + eps)
    # second moment
    x = src.unsqueeze(-1)                                 # [N, D, 1]
    xxT = x @ x.transpose(1, 2)                           # [N, D, D]
    xxT_flat = xxT.reshape(N, D*D)                        # [N, D^2]
    sum_xxT_flat = scatter_add(xxT_flat, index, dim=0, dim_size=G)  # [G, D^2]
    E_xxT = sum_xxT_flat.reshape(G, D, D) / (counts[:, None, None] + eps)
    # covariance
    mu_muT = mu.unsqueeze(-1) @ mu.unsqueeze(1)           # [G, D, D]
    cov = E_xxT - mu_muT
    cov = cov + eps * torch.eye(D, device=src.device).unsqueeze(0)
    # eigendecompose
    eigvals, eigvecs = torch.linalg.eigh(cov)             # eigvals [G, D], eigvecs [G, D, D]
    return eigvals, eigvecs

# def compute_features(pos: np.ndarray, colors: np.ndarray, r=0.05, eps=1e-6):
#     """Compute per-point geometric + color features."""
#     pos_t = torch.from_numpy(pos).float()
#     # build k-NN for local neighborhoods
#     # edge_index = knn_graph(pos_t, k=knn_k)  # [2, E]
#     edge_index = radius_graph(pos_t, r=r)
#     src_idx = edge_index[0]
#     nbr_idx = edge_index[1]
#     # gather neighbor coordinates
#     src = pos_t[nbr_idx]  # [E, 3]
#     # scatter-eigendecompose: groups src by src_idx (center index)
#     eigvals, eigvecs = scatter_eigendecomposition(src, src_idx, G=pos_t.size(0), eps=eps)
#     # eigenvalues sorted ascending: l1 ≤ l2 ≤ l3
#     l1, l2, l3 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
#     # geometric features
#     linearity  = (l3 - l2) / (l3 + eps)
#     planarity  = (l2 - l1) / (l3 + eps)
#     scattering = l1 / (l3 + eps)
#     # verticality: abs of normal's Z component (normal = eigenvector of l1)
#     normals = eigvecs[:, :, 0]  # [N, 3]
#     verticality = normals[:, 2].abs()
#     # color features normalized to [0,1]
#     color_t = torch.from_numpy(colors).float() / 255.0  # [N, 3]
#     # stack all features: [N, 7]
#     feats = torch.stack([linearity, planarity, scattering, verticality,
#                          color_t[:, 0], color_t[:, 1], color_t[:, 2]], dim=1)
#     return feats.cpu().numpy()

def build_csr(edge_idx, num_nodes):
    sys.path.append("./segmentation/cutpursuit/pcd-prox-split/grid-graph/python")
    from grid_graph import edge_list_to_forward_star # type: ignore
    src_csr, tgt, reidx = edge_list_to_forward_star(num_nodes, edge_idx.T.contiguous().cpu().numpy())
    return src_csr.astype(np.uint32), tgt.astype(np.uint32), reidx

class Graph:
    def __init__(self, super:NDArray, edges:Tuple[NDArray,NDArray], feats:NDArray) -> None:

        assert feats.shape[0]+1 == edges[0].shape[0]
        self.super = super
        self.edges = edges
        self.feats = feats
        
        
    
    @staticmethod
    def create_cp(
        pos: NDArray,                   # [N,3]
        x: NDArray,                     # [N,F]
        regularization: float   = 0.1,
        spatial_weight: float   = 1.5,
        cutoff: float           = 5,   # minimum superpoint size,
        k: int = 10,

        ) -> "Graph":
        
        n_dim = 3
        n_feat = x.shape[1]
        
        pos_t = torch.from_numpy(pos).float()
        x_t = torch.from_numpy(x).float()
        edge = knn_graph(pos_t, k=10)
        edge = torch.cat([edge, edge.flip(0)], dim=1)
        
        src, tgt, re = build_csr(edge, pos.shape[0])
        ew = np.ones_like(tgt, dtype = np.float64) * regularization
        vw = np.ones(pos.shape[0], dtype = np.float64)
        cw = np.ones(n_dim+n_feat, dtype = np.float64)
        cw[:n_dim] *= spatial_weight
        
        X = np.concatenate([pos - pos.mean(0), x], axis=1)
        Xf = np.asfortranarray(X.T, dtype=np.float64)
        
        print(f'DEBUG creating graph with {x.shape} features')
        super0, xc, clst, edges0, times = cp_d0_dist(
            n_dim + n_feat, Xf, src, tgt,
            edge_weights=ew,
            vert_weights=vw,
            coor_weights=cw,
            min_comp_weight=cutoff,
            cp_dif_tol=1e-2,
            cp_it_max=10,
            split_damp_ratio=0.7, verbose=False,
            max_num_threads=0,
            balance_parallel_split=True,
            compute_Time=True,
            compute_List=True,
            compute_Graph=True
        )
        
        # extract superpoint features
        R = super0.max()+1
        super0_t = torch.as_tensor(super0, dtype = torch.int64)
        super_centroids = scatter_mean(pos_t,super0_t,dim=0,dim_size=R)
        pos_centered = torch.as_tensor(pos_t - super_centroids[super0],dtype=torch.float32)
        shp_evals, shp_evecs = scatter_eigendecomposition(
            pos_centered,
            super0_t,
            R
        )
        l1, l2, l3 = shp_evals[:,0], shp_evals[:,1], shp_evals[:,2]
        
        normals = shp_evecs[:,:,0]
        verticality = normals[:,2].abs()
        eps = 1e-5
        linearity  = (l3 - l2) / (l3 + eps)
        planarity  = (l2 - l1) / (l3 + eps)
        scattering = l1 / (l3 + eps)
        
        mean_feat = scatter_mean(x_t, super0_t, dim=0, dim_size=R)
        graph_embedding = graph_distance_embedding(super_centroids.cpu().numpy(),(edges0[0], edges0[1]),dim=16)

        
        super_feat = torch.cat([
            mean_feat,
            super_centroids,
            normals,
            verticality[:,None],
            linearity[:,None],
            planarity[:,None],
            scattering[:,None]
        ],dim=1).numpy()
        super_feat -= super_feat.mean(axis=0)
        super_feat /= np.std(super_feat,axis=0)
        super_feat = np.concat([super_feat,graph_embedding],axis=1)*10
        
        return Graph(super0, (edges0[0], edges0[1]), feats = super_feat)
        
def compute_point_feats(pos:NDArray,
                        colors:NDArray|None=None,
                        intensities:NDArray|None=None,
                        radius:float=0.05,
                        device:str='cuda',
                        )->NDArray:
    pos_t = torch.from_numpy(pos).float().to(device=device)
    col_t = torch.from_numpy(colors).float().to(device=device) if colors is not None else None
    int_t = torch.from_numpy(intensities).float().to(device=device) if intensities is not None else None
    src_idx, nbr_idx = radius_graph(pos_t, radius)
    src = pos_t[nbr_idx] # [E,3]
    evals, evecs = scatter_eigendecomposition(src, src_idx)
    l1,l2,l3 = evals[:,0],evals[:,1],evals[:,2]
    eps = 1e-6
    linearity  = (l3 - l2) / (l3 + eps)
    planarity  = (l2 - l1) / (l3 + eps)
    scattering = l1 / (l3 + eps)
    
    normals = evecs[:,:,0]
    verticality = normals[:,2].abs()

    feats = torch.stack([
        linearity,planarity,scattering,
        normals[:,0],normals[:,1],normals[:,2],
        verticality
    ],dim=1)
    return feats.cpu().numpy()

def smooth_feat(pos:NDArray,
                feat:NDArray,
                k=5,
                sigma_spatial:float = 0.05,
                device='cuda')->NDArray:
    pos_t = torch.from_numpy(pos).to(device=device)
    feat_t = torch.from_numpy(feat).to(device=device)
    if len(feat.shape) == 1 : feat = feat[:,None]
    src, nbr = knn_graph(pos_t, k)
    color_smooth = scatter_mean(feat_t[nbr], src, dim=0, dim_size=pos_t.size(0))
    diff_pos = (pos_t[src] - pos_t[nbr]).pow(2).sum(1)
    diff_color = (feat_t[src] - feat_t[nbr]).pow(2).sum(1)

    w = torch.exp(-diff_pos/(2*sigma_spatial**2)) * torch.exp(-diff_color/(2*sigma_spatial**2))
    color_smooth = scatter_sum(feat_t[nbr]*w[:,None], src, dim=0)
    color_smooth = color_smooth / (scatter_sum(w, src, dim=0)[:,None] + 1e-8)
    return color_smooth.cpu().numpy()


def create_cp_graph(pos:NDArray, 
                    scalar_fields:Dict,
                    name_exceptions:List[str]=
                    [
                        # "RGB",
                        "scalar_Original_cloud_index"
                    ])-> Graph:
    pos_centered = pos - pos.mean(0)
    print(f'DEBUG: creating graph')
    sfs = []
    # normalize scalar fields
        
    for sn, sf in scalar_fields.items():
        if sn in name_exceptions:
            continue
        assert isinstance(sf, np.ndarray)
        sf = sf.copy().astype(float)
        if len(sf.shape)==1:
            sf = sf[:,None]
        if len(sf.shape)>2 and sf.size != 0:
            sf = sf.reshape(sf.shape[0],-1)            
        sf -= np.mean(sf, axis=0)
        sf /= np.std(sf, axis=0)
        if sn =='RGB':
            sf = smooth_feat(pos, sf)
            sf *= .6
            # print(f'DEBUG smoothing {sn}')
        print(f"DEBUG scalar field data: {sn=}, {sf.shape=}")
        sfs.append(sf)
        print(f"DEBUG cp feat stat {sn}: {sf.min()=}, {sf.max()}")
    field_feats = np.concat(sfs, axis=1)   
    
    # compute neighbor features
    nb_feats = compute_point_feats(pos_centered)
    feats = np.concat([field_feats, nb_feats],axis=1)
    return Graph.create_cp(pos_centered, feats)

class GraphCutDPGMM:
    def __init__(self, graph:Graph, smoothing_beta=.01, pairwise_lambda=10.0,
                 dp_alpha=0.5, max_components=10):
        assert graph.feats is not None
        self.graph = graph
        self.V, self.F = graph.feats.shape
        self.first_edge, self.adj_vertices = graph.edges
        self.beta =smoothing_beta
        self.lmbda = pairwise_lambda
        self.max_components = max_components
        
        self.pos_inds = []
        self.neg_inds = []

        common_kwargs = dict(
            weight_concentration_prior_type = 'dirichlet_process',
            weight_concentration_prior=dp_alpha,
            n_components=max_components,
            covariance_type='full',
            max_iter=100,
            random_state=0
        )
        self.model_fg = BayesianGaussianMixture(**common_kwargs) # type:ignore
        self.model_bg = BayesianGaussianMixture(**common_kwargs) # type:ignore
        n_init = min(self.V, max_components)
        # init_indices = np.random.RandomState(0).choice(self.V, size=n_init, replace=False)
        # dummy = self.features[init_indices]
        # self.model_fg.fit(dummy)
        # self.model_bg.fit(dummy)
    def register_pick(self, pid:int, positive:bool):
        super_id = self.graph.super[pid]
        print(f"DEBUG {super_id=}")
        if positive:
            self.pos_inds.append(super_id)
        else:
            self.neg_inds.append(super_id)
            
        self._update_clicks()
        labels = self._solve()[self.graph.super]
        return labels
        
    def _update_clicks(self):
        assert self.graph.feats is not None
        if len(self.pos_inds) >= 2:
            Xp = self.graph.feats[self.pos_inds]
            k_fg = min(len(self.pos_inds), self.max_components)
            self.model_fg.set_params(n_components=k_fg)
            self.model_fg.fit(Xp)

        if len(self.neg_inds) >= 2:
            Xn = self.graph.feats[self.neg_inds]
            k_bg = min(len(self.neg_inds), self.max_components)
            self.model_bg.set_params(n_components=k_bg)
            self.model_bg.fit(Xn)

    def compute_unary(self):
        assert self.graph.feats is not None
        if len(self.pos_inds) < 2 or len(self.neg_inds) < 2:
            return np.ones(self.graph.feats.shape[0]), np.ones(self.graph.feats.shape[0])
        
        logp_fg = self.model_fg.score_samples(self.graph.feats)
        logp_bg = self.model_bg.score_samples(self.graph.feats)
        cost_source = -logp_fg  # cost source[u]    = -log P(x_u | fg)
        cost_sink   = -logp_bg  # cost_sink[u]      = -log P(x_u | bg)

        return cost_source, cost_sink
    
    def build_graph(self, cost_source, cost_sink):
        assert self.graph.feats is not None
        g = maxflow.Graph[float](self.V, self.adj_vertices.size)
        nodeids = g.add_nodes(self.V)
        # t-links
        for u in range(self.V):
            g.add_tedge(u, cost_source[u], cost_sink[u])
        # pairwise edges
        fe, av = self.first_edge, self.adj_vertices
        print(f"DEBUG {fe.shape=}")
        print(f"DEBUG {self.V=}")
        for u in range(self.V):
            start, end = fe[u], fe[u+1]
            fu = self.graph.feats[u]
            for idx in range(start, end):
                v = av[idx]
                # weight = lambda * exp(-beta * ||f_u - f_v||^2)
                diff = fu - self.graph.feats[v]
                w = self.lmbda * np.exp(-self.beta * (diff @ diff))
                g.add_edge(u,v,w,w)
        return g
    
    def _solve(self):
        cs, ct = self.compute_unary()
        g = self.build_graph(cs, ct)
        flow = g.maxflow()
        print(f"solved maxflow with flow={flow}")
        labels = np.array([g.get_segment(u) for u in range(self.V)])
        return labels.astype(bool)
    

    
    

    
    
    
    