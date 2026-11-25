from mvcha_dp.data.similarities import cached_similarity
import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances
from mvcha_dp.graph.numeric import _to_numpy_float32


def validate_kappa(kappa):
    if not isinstance(kappa, (tuple, list)):
        raise ValueError(f'kappa must be a tuple or list, got {type(kappa).__name__}.')
    if len(kappa) != 3:
        raise ValueError(f'kappa must contain exactly three integers, got length {len(kappa)}.')
    if not all((isinstance(x, int) for x in kappa)):
        raise ValueError(f'all kappa values must be integers, got {kappa}.')
    if not all((x > 0 for x in kappa)):
        raise ValueError(f'all kappa values must be positive, got {kappa}.')
    if not kappa[0] < kappa[1] < kappa[2]:
        raise ValueError(f'kappa must be strictly increasing, e.g., (4, 5, 6), got {kappa}.')


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


def compute_comprehensive_similarity(X_struct, importance_scores=None, alpha=0.7, use_node_importance=True):
    sim_struct = compute_structural_similarity(X_struct)
    if use_node_importance and importance_scores is not None:
        sim_imp = compute_importance_similarity(importance_scores)
        sim = alpha * sim_struct + (1.0 - alpha) * sim_imp
    else:
        sim = sim_struct
    return sim.astype(np.float32)


def high_similarity_tail_fraction(cfg):
    new = cfg.get('high_similarity_tail_fraction')
    old = cfg.get('dynamic_k_quantile', cfg.get('q'))
    if new is not None and old is not None and (float(new) != float(old)):
        raise ValueError('Conflicting tail fraction values')
    return float(new if new is not None else old if old is not None else 0.2)


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


def choose_dynamic_k(sim_row, eta, tau1=0.2, tau2=0.5, k_values=None, kappa=None):
    if kappa is not None:
        k_vals = kappa
    elif k_values is not None:
        k_vals = k_values
    else:
        k_vals = (4, 5, 6)
    sim_row = _to_numpy_float32(sim_row).reshape(-1)
    valid = sim_row[sim_row >= 0.0]
    if valid.size == 0:
        return int(k_vals[0])
    high_ratio = float(np.mean(valid >= eta))
    if high_ratio >= tau2:
        return int(k_vals[2])
    if high_ratio >= tau1:
        return int(k_vals[1])
    return int(k_vals[0])


def build_view_hyperedges(X_view, importance_scores, cfg, return_similarity=False):
    alpha = cfg.get('alpha', 0.7)
    tau1 = cfg.get('tau1', 0.2)
    tau2 = cfg.get('tau2', 0.5)
    k_values = cfg.get('kappa', cfg.get('k_values', (4, 5, 6)))
    q = high_similarity_tail_fraction(cfg)
    weight_mode = cfg.get('weight_mode', 'unit')
    use_node_importance = cfg.get('use_node_importance', True)
    use_dynamic_k = cfg.get('use_dynamic_k', True)
    fixed_k = cfg.get('fixed_k', 5)
    sim = compute_comprehensive_similarity(
        X_view,
        importance_scores=importance_scores,
        alpha=alpha,
        use_node_importance=use_node_importance
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
        if use_dynamic_k:
            k_i = choose_dynamic_k(sim_i, eta=eta, tau1=tau1, tau2=tau2, k_values=k_values)
        else:
            k_i = int(fixed_k)
        other = np.arange(n)[np.arange(n) != i]
        nbr_idx = other[np.argsort(-sim_i[other], kind='stable')[:min(k_i, n - 1)]]
        edge = sorted(set([i] + nbr_idx.tolist()))
        hyperedges.append(edge)
        if weight_mode == 'unit':
            w = 1.0
        elif weight_mode == 'mean_similarity':
            sub_sim = sim[np.ix_(edge, edge)]
            w = float(sub_sim.mean())
        else:
            raise ValueError(f'Unsupported weight_mode: {weight_mode}')
        weights.append(w)
    if return_similarity:
        return (hyperedges, weights, sim)
    return (hyperedges, weights)


def fuse_multi_view_hyperedges(view_hyperedges, view_weights):
    (all_edges, all_weights) = ([], [])
    for (edges, weights) in zip(view_hyperedges, view_weights):
        all_edges.extend(edges)
        all_weights.extend(weights)
    return (all_edges, all_weights)


def build_mvcha_hypergraph(view_features, X_concat, E, cfg=None, device=None, precomputed_importance=None, dependency_template=None):
    if cfg is None:
        cfg = {}
    from mvcha_dp.protocol.contract import is_parity
    parity = is_parity(cfg)
    if not isinstance(view_features, dict) or set(view_features) != {'TSM', 'CNM', 'NEM'}:
        raise ValueError('Expected named TSM, CNM and NEM views')
    if isinstance(view_features, dict):
        view_features = [view_features[k] for k in ('TSM', 'CNM', 'NEM')]
    X_concat = _to_numpy_float32(X_concat)
    num_nodes = X_concat.shape[0]
    if len(view_features) != 3:
        raise ValueError('Expected three feature views')
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
    fusion = None
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
