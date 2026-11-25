from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DependencyGraph:
    num_nodes: int
    positive_pairs: tuple
    neighbors: tuple
    observed_self_loops: tuple
    message_source: torch.Tensor
    message_destination: torch.Tensor
    message_weight: torch.Tensor
    directed: bool = False

    @classmethod
    def from_edges(cls, num_nodes, edges, directed=False):
        if num_nodes < 1:
            raise ValueError('num_nodes must be positive')
        (pairs, loops, raw_pairs) = (set(), set(), set())
        for pair in edges:
            if len(pair) != 2:
                raise ValueError('Expected (u,v) edge rows, not an adjacency matrix')
            (u, v) = (int(pair[0]), int(pair[1]))
            if u != pair[0] or v != pair[1] or (not (0 <= u < num_nodes and 0 <= v < num_nodes)):
                raise ValueError('Invalid dependency node index')
            if u == v:
                loops.add(u)
            else:
                pairs.add((min(u, v), max(u, v)))
                raw_pairs.add((u, v) if directed else (min(u, v), max(u, v)))
        pairs = tuple(sorted(pairs))
        adjacent = [set() for _ in range(num_nodes)]
        for (u, v) in pairs:
            adjacent[u].add(v)
            adjacent[v].add(u)
        degree = [len(adjacent[i]) + 1 + int(i in loops) for i in range(num_nodes)]
        messages = [(u, v, 1.0) for (u, v) in pairs] + [(v, u, 1.0) for (u, v) in pairs]
        messages += [(i, i, float(1 + int(i in loops))) for i in range(num_nodes)]
        source = torch.tensor([u for (u, _, _) in messages], dtype=torch.long)
        destination = torch.tensor([v for (_, v, _) in messages], dtype=torch.long)
        weights = torch.tensor([w / math.sqrt(degree[u] * degree[v]) for (u, v, w) in messages], dtype=torch.float32)
        return cls(
            num_nodes,
            tuple(sorted(raw_pairs)),
            tuple((frozenset(a) for a in adjacent)),
            tuple(sorted(loops)),
            source,
            destination,
            weights,
            bool(directed)
        )

    def propagate(self, features):
        if features.ndim != 2 or features.shape[0] != self.num_nodes:
            raise ValueError('Features must have one row per real node')
        src = self.message_source.to(features.device)
        dst = self.message_destination.to(features.device)
        weight = self.message_weight.to(device=features.device, dtype=features.dtype)
        result = torch.zeros_like(features)
        return result.index_add(0, dst, features[src] * weight[:, None])


class ImportanceEncoder(nn.Module):

    def __init__(self, input_dim=69, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.gcn1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, graph):
        h = F.relu(self.gcn1(graph.propagate(features)))
        h = self.dropout(h)
        return h


class ImportanceHead(nn.Module):

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings):
        return self.linear(embeddings).squeeze(-1)


class BilinearStructureDecoder(nn.Module):

    def __init__(self, hidden_dim=64, directed=False):
        super().__init__()
        self.B = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(()))
        self.directed = bool(directed)
        nn.init.xavier_uniform_(self.B)

    def forward(self, embeddings, pairs):
        pairs = pairs.to(device=embeddings.device, dtype=torch.long)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError('pairs must have shape [P,2]')
        B = self.B if self.directed else (self.B + self.B.T) / 2
        return (embeddings[pairs[:, 0]] @ B * embeddings[pairs[:, 1]]).sum(dim=1) + self.bias


class V2NodeImportanceModel(nn.Module):

    def __init__(self, input_dim=69, hidden_dim=64, dropout=0.3, directed=False, score_bound_b=2.0):
        super().__init__()
        self.encoder = ImportanceEncoder(input_dim, hidden_dim, dropout)
        self.head = ImportanceHead(hidden_dim)
        self.decoder = BilinearStructureDecoder(hidden_dim, directed)
        if not math.isfinite(score_bound_b) or score_bound_b <= 0:
            raise ValueError('score_bound_b must be finite and positive')
        self.score_bound_b = float(score_bound_b)

    def forward(self, features, graph, tau0=1.0):
        if not math.isfinite(tau0) or tau0 <= 0:
            raise ValueError('tau0 must be finite and positive')
        h = self.encoder(features, graph)
        r = self.head(graph.propagate(h))
        scores = self.score_bound_b * tau0 * torch.tanh(r / (self.score_bound_b * tau0))
        return dict(
            h_imp=h,
            pre_score_r=r,
            bounded_score_s=scores,
            scores=scores,
            p=torch.softmax(scores / tau0, dim=0)
        )


def k_hop_context_nodes(graph, candidate, context_hops):
    if not 0 <= candidate < graph.num_nodes or context_hops < 1:
        raise ValueError('Invalid candidate or context_hops')
    (visited, frontier) = ({candidate}, {candidate})
    for _ in range(context_hops):
        new = {v for u in frontier for v in graph.neighbors[u]} - visited
        visited.update(new)
        frontier = new
        if not frontier:
            break
    return tuple(sorted(visited - {candidate}))


def _sample_non_edges(nodes, positives, count, generator, directed=False):
    available = len(nodes) * (len(nodes) - 1) // (1 if directed else 2) - len(positives)
    count = min(count, available)
    chosen = set()
    if count == 0:
        return ([], available, False)
    for _ in range(max(64, 10 * count)):
        idx = torch.randint(len(nodes), (2,), generator=generator).tolist()
        (u, v) = (nodes[idx[0]], nodes[idx[1]])
        pair = (u, v) if directed else (min(u, v), max(u, v))
        if u != v and pair not in positives:
            chosen.add(pair)
        if len(chosen) == count:
            return (sorted(chosen), available, False)
    (need, reservoir, seen) = (count - len(chosen), [], 0)
    universe = ((u, v) for u in nodes for v in nodes if u != v) if directed else combinations(nodes, 2)
    for pair in universe:
        if pair in positives or pair in chosen:
            continue
        seen += 1
        if len(reservoir) < need:
            reservoir.append(pair)
        else:
            position = int(torch.randint(seen, (), generator=generator))
            if position < need:
                reservoir[position] = pair
    return (sorted(chosen | set(reservoir)), available, True)


def sample_dependency_pairs(graph, generator, negative_ratio=1.0, max_positive_samples=None, context_nodes=None):
    if not math.isfinite(negative_ratio) or negative_ratio < 0:
        raise ValueError('negative_ratio must be finite and nonnegative')
    if max_positive_samples is not None and max_positive_samples < 1:
        raise ValueError('max_positive_samples must be positive')
    nodes = tuple(range(graph.num_nodes)) if context_nodes is None else tuple(sorted(set(context_nodes)))
    if any((u < 0 or u >= graph.num_nodes for u in nodes)):
        raise ValueError('Invalid context node')
    node_set = set(nodes)
    positive_universe = {(u, v) for (u, v) in graph.positive_pairs if u in node_set and v in node_set}
    positive = sorted(positive_universe)
    if max_positive_samples is not None and len(positive) > max_positive_samples:
        indices = torch.randperm(len(positive), generator=generator)[:max_positive_samples].tolist()
        positive = [positive[i] for i in indices]
    requested = math.ceil(len(positive) * negative_ratio)
    if max_positive_samples is not None:
        requested = min(requested, max_positive_samples)
    (negative, available, fallback) = _sample_non_edges(nodes, positive_universe, requested, generator, graph.directed)
    pairs = torch.tensor(positive + negative, dtype=torch.long).reshape(-1, 2)
    adj_target = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))])
    info = dict(
        positive_count=len(positive),
        negative_count=len(negative),
        requested_negatives=requested,
        available_negatives=available,
        negative_shortfall=requested - len(negative),
        reservoir_fallback=fallback,
        reason='no_positive_pairs' if not positive else 'no_negative_pairs' if not negative else None
    )
    return dict(pairs=pairs, adj_target=adj_target, diagnostics=info)


def build_context_pairs(graph, candidate, context_hops, generator, negative_ratio=1.0, max_positive_samples=None):
    context = k_hop_context_nodes(graph, candidate, context_hops)
    sampled = sample_dependency_pairs(graph, generator, negative_ratio, max_positive_samples, context)
    sampled['context_nodes'] = context
    assert candidate not in sampled['pairs'].reshape(-1).tolist()
    return sampled


def perturb_node_input(features, candidate):
    if not 0 <= candidate < len(features):
        raise ValueError('Invalid candidate')
    perturbed = features.clone()
    perturbed[candidate] = 0
    return perturbed


def reconstruction_loss(embeddings, pairs, adj_target, decoder):
    if len(pairs) != len(adj_target):
        raise ValueError('Pairs and adjacency targets must align')
    if not len(pairs) or not bool((adj_target == 1).any()) or (not bool((adj_target == 0).any())):
        return embeddings.sum() * 0.0
    target = adj_target.to(embeddings)
    values = F.binary_cross_entropy_with_logits(decoder(embeddings, pairs), target, reduction='none')
    return 0.5 * values[target == 1].mean() + 0.5 * values[target == 0].mean()


def neighbor_embedding_distortion(normal, perturbed, context_nodes, cosine_eps=1e-12):
    if not len(context_nodes) or cosine_eps <= 0:
        raise ValueError('Nonempty context and positive cosine epsilon required')
    idx = torch.tensor(context_nodes, dtype=torch.long, device=normal.device)
    (a, b) = (normal[idx], perturbed[idx])
    (na, nb) = (a.norm(dim=1), b.norm(dim=1))
    (za, zb) = (na <= cosine_eps, nb <= cosine_eps)
    cosine = ((a * b).sum(1) / (na * nb).clamp_min(cosine_eps ** 2)).clamp(-1, 1)
    difference = torch.where(za & zb, torch.zeros_like(na), torch.where(za | zb, torch.ones_like(na), 1 - cosine))
    return difference.mean()


def normalize_contribution_signal(delta, epsilon=0.0001, clip=3.0):
    if not math.isfinite(epsilon) or epsilon <= 0 or (not math.isfinite(clip)) or (clip <= 0):
        raise ValueError('Positive finite normalization epsilon/clip required')
    if delta.ndim != 1 or not torch.isfinite(delta).all():
        raise ValueError('Expected finite 1-D signal')
    if not delta.numel():
        return torch.zeros_like(delta)
    std = delta.std(unbiased=False)
    if bool(std <= epsilon):
        return torch.zeros_like(delta)
    return ((delta - delta.mean()) / (std + epsilon)).clamp(-clip, clip)


def pairwise_ranking_loss(scores, contribution, contribution_tie_margin=1e-06):
    if scores.ndim != 1 or scores.shape != contribution.shape:
        raise ValueError('Aligned one-dimensional scores/contribution required')
    if not math.isfinite(contribution_tie_margin) or contribution_tie_margin < 0:
        raise ValueError('Invalid tie margin')
    total = len(scores) * (len(scores) - 1) // 2
    if len(scores) < 2:
        return (
            scores.sum() * 0.0,
            dict(num_valid_pairs=0, total_pairs=total, tie_ratio=0.0, reason='insufficient_valid_candidates')
        )
    pairs = torch.triu_indices(len(scores), len(scores), offset=1, device=scores.device)
    c = contribution.detach()
    delta = c[pairs[0]] - c[pairs[1]]
    valid = (delta.abs() >= contribution_tie_margin) & (delta != 0)
    count = int(valid.sum())
    info = dict(
        num_valid_pairs=count,
        total_pairs=total,
        tie_ratio=1 - count / total,
        reason=None if count else 'all_ties'
    )
    if count == 0:
        return (scores.sum() * 0.0, info)
    (a, b) = pairs[:, valid]
    loss = F.softplus(-delta[valid].sign() * (scores[a] - scores[b])).mean()
    return (loss, info)


@contextmanager
def _evaluation_mode(model):
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for (module, flag) in states:
            module.training = flag


def _statistics(values):
    if not values.numel():
        return dict(count=0, min=None, max=None, mean=None, std=None, negative_ratio=None)
    v = values.detach()
    return dict(
        count=v.numel(),
        min=float(v.min()),
        max=float(v.max()),
        mean=float(v.mean()),
        std=float(v.std(unbiased=False)),
        negative_ratio=float((v < 0).float().mean())
    )


def compute_contextual_contributions(model, features, graph, *, generator, sample_size=16, context_hops=2, rho=0.5, tau0=1.0, context_pairs_per_class=64, norm_std_floor_rec=0.0001, norm_std_floor_emb=0.0001, norm_clip=3.0, zero_norm_eps=1e-12):
    if sample_size < 1 or context_hops != 2 or (not 0 <= rho <= 1):
        raise ValueError('Invalid importance sampling configuration')
    eligible = []
    for i in range(graph.num_nodes):
        nodes = k_hop_context_nodes(graph, i, context_hops)
        ns = set(nodes)
        pos = sum((u in ns and v in ns for (u, v) in graph.positive_pairs))
        total = len(nodes) * (len(nodes) - 1) // (1 if graph.directed else 2)
        if len(nodes) >= 2 and 0 < pos < total:
            eligible.append(i)
    order = torch.randperm(len(eligible), generator=generator)[:min(sample_size, len(eligible))].tolist()
    candidates = [eligible[k] for k in order]
    (rec, emb, contexts) = ([], [], [])
    with _evaluation_mode(model), torch.no_grad():
        reference = model(features, graph, tau0)
        for i in candidates:
            sampled = build_context_pairs(graph, i, context_hops, generator, 1.0, context_pairs_per_class)
            h_masked = model.encoder(perturb_node_input(features, i), graph)
            normal = reconstruction_loss(reference['h_imp'], sampled['pairs'], sampled['adj_target'], model.decoder)
            masked = reconstruction_loss(h_masked, sampled['pairs'], sampled['adj_target'], model.decoder)
            rec.append(masked - normal)
            emb.append(neighbor_embedding_distortion(
                reference['h_imp'],
                h_masked,
                sampled['context_nodes'],
                zero_norm_eps
            ))
            contexts.append(dict(
                candidate=i,
                context_nodes=list(sampled['context_nodes']),
                pairs=sampled['pairs'].tolist(),
                labels=sampled['adj_target'].tolist(),
                **sampled['diagnostics']
            ))
    dr = torch.stack(rec) if rec else features.new_empty(0)
    de = torch.stack(emb) if emb else features.new_empty(0)
    nr = normalize_contribution_signal(dr, norm_std_floor_rec, norm_clip)
    ne = normalize_contribution_signal(de, norm_std_floor_emb, norm_clip)
    contribution = (rho * nr + (1 - rho) * ne).detach()
    return dict(
        candidate_nodes=torch.tensor(candidates, dtype=torch.long, device=features.device),
        delta_rec=dr,
        delta_emb=de,
        contribution=contribution,
        contexts=contexts,
        eligible_candidate_count=len(eligible),
        normalized_rec=nr,
        normalized_emb=ne
    )


def compute_v2_self_supervised_loss(model, features, graph, *, generator, sample_size=16, context_hops=2, rho=0.5, tau0=1.0, negative_ratio=1.0, lambda_rec=1.0, lambda_imp=1.0, contribution_tie_margin=1e-06, max_positive_samples=None, context_pairs_per_class=64, norm_std_floor_rec=0.0001, norm_std_floor_emb=0.0001, norm_clip=3.0, zero_norm_eps=1e-12, ranking_enabled=True):
    if not all((math.isfinite(v) and v >= 0 for v in (lambda_rec, lambda_imp))):
        raise ValueError('Nonnegative finite loss weights required')
    if not 0 <= rho <= 1:
        raise ValueError('rho must be in [0,1]')
    if ranking_enabled:
        context = compute_contextual_contributions(
            model,
            features,
            graph,
            generator=generator,
            sample_size=sample_size,
            context_hops=context_hops,
            rho=rho,
            tau0=tau0,
            context_pairs_per_class=context_pairs_per_class,
            norm_std_floor_rec=norm_std_floor_rec,
            norm_std_floor_emb=norm_std_floor_emb,
            norm_clip=norm_clip,
            zero_norm_eps=zero_norm_eps
        )
    else:
        empty = features.new_empty(0)
        context = dict(
            candidate_nodes=torch.empty(0, dtype=torch.long, device=features.device),
            delta_rec=empty,
            delta_emb=empty,
            contribution=empty,
            contexts=[],
            eligible_candidate_count=0,
            normalized_rec=empty,
            normalized_emb=empty
        )
    sampled = sample_dependency_pairs(graph, generator, negative_ratio, max_positive_samples)
    training = model(features, graph, tau0)
    loss_rec = reconstruction_loss(training['h_imp'], sampled['pairs'], sampled['adj_target'], model.decoder)
    ids = context['candidate_nodes']
    (loss_imp, ranking) = pairwise_ranking_loss(
        training['bounded_score_s'][ids],
        context['contribution'],
        contribution_tie_margin
    )
    terms = []
    if lambda_rec > 0 and sampled['diagnostics']['reason'] is None:
        terms.append(lambda_rec * loss_rec)
    if lambda_imp > 0 and ranking_enabled and (ranking['num_valid_pairs'] > 0):
        terms.append(lambda_imp * loss_imp)
    loss_self = sum(terms) if terms else training['bounded_score_s'].sum() * 0.0
    diagnostics = dict(
        reconstruction=dict(**sampled['diagnostics'], loss=float(loss_rec.detach())),
        sampled_count=len(ids),
        valid_count=len(ids),
        skipped_count=graph.num_nodes - context['eligible_candidate_count'],
        eligible_candidate_count=context['eligible_candidate_count'],
        contexts=context['contexts'],
        skip_reasons={'insufficient_context_or_missing_pair_class': graph.num_nodes - context['eligible_candidate_count']},
        delta_rec=_statistics(context['delta_rec']),
        delta_emb=_statistics(context['delta_emb']),
        normalized_rec=_statistics(context['normalized_rec']),
        normalized_emb=_statistics(context['normalized_emb']),
        contribution=_statistics(context['contribution']),
        ranking=ranking,
        global_forward_mode='train' if model.training else 'eval',
        contribution_forward_mode='eval_no_grad',
        ranking_score_forward_mode='train' if model.training else 'eval',
        masked_requires_grad=[False] * len(ids),
        no_dense_reconstruction=True,
        has_optimization_signal=bool(terms)
    )
    return dict(
        loss_rec=loss_rec,
        loss_imp=loss_imp,
        loss_self=loss_self,
        scores=training['bounded_score_s'],
        bounded_score_s=training['bounded_score_s'],
        pre_score_r=training['pre_score_r'],
        p=training['p'],
        candidate_nodes=ids,
        valid_candidate_nodes=ids,
        delta_rec=context['delta_rec'],
        delta_emb=context['delta_emb'],
        contribution=context['contribution'],
        num_valid_pairs=ranking['num_valid_pairs'],
        diagnostics=diagnostics
    )
