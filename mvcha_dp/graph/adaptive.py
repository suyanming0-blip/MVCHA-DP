from mvcha_dp.data.similarities import cached_similarity
import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances
from mvcha_dp.graph.numeric import _to_numpy_float32


def compute_structural_similarity(X_struct):
    cached = cached_similarity(X_struct)
    if cached is not None:
        return cached
    X_np = _to_numpy_float32(X_struct)
    dist = pairwise_distances(X_np, metric='euclidean')
    values = dist[np.triu_indices(len(dist), 1)]
    if not len(values):
        return np.zeros_like(dist, dtype=np.float32)
    (d_min, d_max) = (values.min(), values.max())
    span = d_max - d_min
    if span == 0:
        return np.zeros_like(dist, dtype=np.float32)
    return ((d_max - dist) / span).astype(np.float32)


def compute_importance_similarity(importance_scores):
    s = _to_numpy_float32(importance_scores).reshape(-1, 1)
    sim_imp = np.minimum(s, s.T)
    return sim_imp.astype(np.float32)


def compute_comprehensive_similarity(X_struct, importance_scores, alpha=0.7):
    sim_struct = compute_structural_similarity(X_struct)
    sim_imp = compute_importance_similarity(importance_scores)
    sim = alpha * sim_struct + (1.0 - alpha) * sim_imp
    return sim.astype(np.float32)


def high_similarity_tail_fraction(cfg):
    return float(cfg['high_similarity_tail_fraction'])


def estimate_global_high_similarity_threshold(sim, q=0.2):
    sim = _to_numpy_float32(sim)
    n = sim.shape[0]
    vals = sim[np.triu_indices(n, 1)]
    if vals.size == 0:
        return 0.0
    if not 0 < q < 1:
        raise ValueError('tail fraction must be in (0,1)')
    eta = np.quantile(vals, 1.0 - q, interpolation='linear')
    return float(eta)


def choose_dynamic_k(sim_row, eta, tau1=0.2, tau2=0.5, kappa=(4, 5, 6)):
    sim_row = _to_numpy_float32(sim_row).reshape(-1)
    valid = sim_row[sim_row >= 0.0]
    if valid.size == 0:
        return int(kappa[0])
    high_ratio = float(np.mean(valid >= eta))
    if high_ratio >= tau2:
        return int(kappa[2])
    if high_ratio >= tau1:
        return int(kappa[1])
    return int(kappa[0])


def build_view_hyperedges(X_view, importance_scores, cfg):
    alpha = cfg.get('alpha', 0.7)
    tau1 = cfg.get('tau1', 0.2)
    tau2 = cfg.get('tau2', 0.5)
    kappa = cfg['kappa']
    q = high_similarity_tail_fraction(cfg)
    sim = compute_comprehensive_similarity(
        X_view,
        importance_scores=importance_scores,
        alpha=alpha
    )
    eta = estimate_global_high_similarity_threshold(sim, q=q) if len(sim) > 1 else None
    hyperedges = []
    weights = []
    n = sim.shape[0]
    for i in range(n):
        if n == 1:
            hyperedges.append([i])
            weights.append(1.0)
            continue
        sim_i = sim[i].copy()
        sim_i[i] = -1.0
        k_i = choose_dynamic_k(sim_i, eta=eta, tau1=tau1, tau2=tau2, kappa=kappa)
        other = np.arange(n)[np.arange(n) != i]
        nbr_idx = other[np.argsort(-sim_i[other], kind='stable')[:min(k_i, n - 1)]]
        edge = sorted(set([i] + nbr_idx.tolist()))
        hyperedges.append(edge)
        weights.append(1.0)
    return (hyperedges, weights)


def fuse_multi_view_hyperedges(view_hyperedges, view_weights):
    (all_edges, all_weights) = ([], [])
    for (edges, weights) in zip(view_hyperedges, view_weights):
        all_edges.extend(edges)
        all_weights.extend(weights)
    return (all_edges, all_weights)


def build_mvcha_hypergraph(view_features, X_concat, E, cfg, precomputed_importance=None, dependency_template=None):
    if not isinstance(view_features, dict) or set(view_features) != {'TSM', 'CNM', 'NEM'}:
        raise ValueError('Expected named TSM, CNM and NEM views')
    view_features = [view_features[k] for k in ('TSM', 'CNM', 'NEM')]
    X_concat = _to_numpy_float32(X_concat)
    num_nodes = X_concat.shape[0]
    if precomputed_importance is not None:
        if torch.is_tensor(precomputed_importance) and precomputed_importance.requires_grad:
            raise ValueError('Graph importance must be detached before injection')
        importance_scores = _to_numpy_float32(precomputed_importance)
        if importance_scores.shape != (num_nodes,) or not np.isfinite(importance_scores).all() or (importance_scores < 0).any() or (not np.isclose(
            importance_scores.sum(),
            1.0,
            rtol=1e-05,
            atol=1e-06
        )):
            raise ValueError('Expected a finite nonnegative normalized N-vector')
    else:
        raise ValueError('A detached runtime importance prior is required')
    view_hyperedges = []
    view_weights = []
    view_names = []
    for (name, X_view) in zip(('TSM', 'CNM', 'NEM'), view_features):
        if X_view is None:
            continue
        view_names.append(name)
        (edges, weights) = build_view_hyperedges(X_view=X_view, importance_scores=importance_scores, cfg=cfg)
        view_hyperedges.append(edges)
        view_weights.append(weights)
    (all_edges, all_weights) = fuse_multi_view_hyperedges(view_hyperedges, view_weights)
    from mvcha_dp.graph.fusion import DependencyTemplate
    template = dependency_template if dependency_template is not None else DependencyTemplate(num_nodes, E)
    (H, W, fusion) = template.fuse(view_names, view_hyperedges)
    return {
        'importance_scores': importance_scores,
        'view_names': view_names,
        'view_hyperedges': view_hyperedges,
        'view_weights': view_weights,
        'hyperedges': all_edges,
        'weights': all_weights,
        'H': H,
        'W': W,
        'fusion': fusion
    }
