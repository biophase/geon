from typing import Dict, Tuple, List
from torch_geometric.nn import knn_graph
from typing import Optional, Tuple
from torch_scatter import scatter_add
import torch
@torch.no_grad()
def scatter_eigendecomposition(src:torch.Tensor, index: torch.Tensor, G:Optional[int]=None, eps:float=1e-6)->Tuple[torch.Tensor, torch.Tensor]:
    """Performs eigendecomposition on subsets of `src` defined by `index`

    Args:
        src (torch.Tensor): Source data [N,D]
        index (torch.Tensor): Index into output [N,]->[0..G]
        G (int): Sets the output size explicitly, defaults to max(index) +1

    Returns:
        Tuple[torch.Tensor,torch.Tensor]: sorted eigenvalues [G,D] and corresponding eigenvectors [G,D,D]
    """
    
    # _, counts = torch.unique(index, return_counts=True)
    if G is None:
        G = int(index.amax().item()) + 1
    
    N, D = src.shape
    counts = torch.bincount(index, minlength=G)
    
    # print(f'old G={int(index.max().item()) + 1}, WhyNotThisG = {counts.size(0)}')
    # if (int(index.max().item()) + 1 != counts.size(0)):
    #     raise Exception


    ones    = torch.ones_like(index, dtype=src.dtype)
    counts  = scatter_add(ones, index, dim=0, dim_size=G)  # [G]



    sum_src = scatter_add(src, index, dim=0, dim_size=G)        # [G,D]
    mu = sum_src / (counts[:,None]+eps)                         # [G,D]

    x = src [...,None]              # [N,D,1]
    xxT = x @ x.permute(0,2,1)      # [N,D,D]
    xxT_flat = xxT.reshape(N,D*D)   # [N,D^2]

    sum_xxT_flat = scatter_add(xxT_flat, index, dim=0, dim_size=G)      # [G,D^2]
    E_xxT = sum_xxT_flat / (counts[:,None]+eps)                         # [G,D,D]
    E_xxT = E_xxT.reshape(G, D, D)


    mu_muT = mu[...,None] @ mu[:,None,:]    # [G,D,1] @ [G,!,D] -> [G,D,D]
    cov = E_xxT - mu_muT                    # [G,D,D]
    
    # Ridge (Tikhonov)-regularization
    eye = torch.eye(D, device=src.device).unsqueeze(0)  # [1,D,D]
    cov = cov + eps * eye

    eigenvals, eigenvecs = torch.linalg.eigh(cov)

    return eigenvals, eigenvecs



def compute_geometric_feats(pos:torch.Tensor,edge_index:torch.Tensor,feat_names: List[str]=[
    'normals',
    'verticality',
    'linearity',
    'planarity',
    'scattering',
    'sum_evals',
    'omnivariance',
    'anisotropy',
    'eigenentropy',
    'sphericity',
    'surface_variation',
    'flatness',
    ])->Dict[str, torch.Tensor]:

    
    ct_i, nb_i = edge_index
    rel_pos_nb = pos[nb_i] - pos[ct_i]

    evals, evecs = scatter_eigendecomposition(rel_pos_nb, ct_i, G = pos.shape[0])
    eps = 1e-5
    l1, l2, l3 = evals[:,0], evals[:,1], evals[:,2]
    normals = evecs[:,:,0]
    verticality = normals[:,2].abs()
    linearity  = (l3 - l2) / (l3 + eps)
    planarity  = (l2 - l1) / (l3 + eps)
    scattering = l1 / (l3 + eps)
    sum_evals = l1 + l2 + l3 + eps
    omnivariance = (l1 * l2 * l3).pow(1/3)
    anisotropy = (l3 - l1) / (l3 + eps)
    p1 = l1 / sum_evals
    p2 = l2 / sum_evals
    p3 = l3 / sum_evals
    eigenentropy = -(p1 * torch.log(p1 + eps) + p2 * torch.log(p2 + eps) + p3 * torch.log(p3 + eps))
    change_curvature = l1 / sum_evals
    sphericity = change_curvature 
    surface_variation = l1 / (l1 + l2 + l3 + eps)
    flatness = l2 / (l3 + eps)
    out = dict()
    if 'normals' in feat_names: out['normals'] = normals
    if 'verticality' in feat_names: out['verticality'] = verticality
    if 'linearity' in feat_names: out['linearity'] = linearity
    if 'planarity' in feat_names: out['planarity'] = planarity
    if 'scattering' in feat_names: out['scattering'] = scattering
    if 'sum_evals' in feat_names: out['sum_evals'] = sum_evals
    if 'omnivariance' in feat_names: out['omnivariance'] = omnivariance
    if 'anisotropy' in feat_names: out['anisotropy'] = anisotropy
    if 'eigenentropy' in feat_names: out['eigenentropy'] = eigenentropy
    if 'sphericity' in feat_names: out['sphericity'] = sphericity
    if 'surface_variation' in feat_names: out['surface_variation'] = surface_variation
    if 'flatness' in feat_names: out['flatness'] = flatness
    return out



