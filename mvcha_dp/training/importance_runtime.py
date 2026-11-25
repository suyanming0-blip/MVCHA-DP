# Node importance updates and hypergraph synchronization.

import copy
import numpy as np
import torch
from mvcha_dp.models.importance import DependencyGraph, V2NodeImportanceModel, compute_v2_self_supervised_loss
from mvcha_dp.graph.adaptive import build_mvcha_hypergraph
DEFAULTS = dict(
    hidden_dim=64,
    dropout=0.3,
    lr=0.001,
    tau0=1.0,
    lambda_rec=1.0,
    lambda_imp=1.0,
    rho=0.5,
    context_hops=2,
    num_candidates=16,
    negative_ratio=1.0,
    contribution_tie_margin=1e-06,
    max_positive_samples=None,
    score_bound_b=2.0,
    dependency_directed=None,
    reconstruction_warmup_steps=10,
    context_pairs_per_class=64,
    norm_std_floor_rec=0.0001,
    norm_std_floor_emb=0.0001,
    norm_clip=3.0,
    zero_norm_eps=1e-12,
    initialization_importance='uniform'
)


def importance_mode(cfg):
    if cfg.get('importance_mode') != 'paper_v2':
        raise ValueError('Expected importance_mode=paper_v2')
    return 'paper_v2'


def uses_v2(cfg):
    return importance_mode(cfg) == 'paper_v2'


def effective_v2_config(cfg):
    importance_mode(cfg)
    return dict(DEFAULTS, **cfg.get('importance_v2', {}))


def parameter_bytes(model):
    return tuple(p.detach().cpu().contiguous().numpy().tobytes() for p in model.parameters())


class V2ImportanceRuntime:

    def __init__(self, X, edges, cfg, device, role='source', identifier='', fold_id=0, node_order=None):
        self.mode = importance_mode(cfg)
        self.config = effective_v2_config(cfg)
        if set(self.config) != set(DEFAULTS):
            raise ValueError('Unknown importance_v2 setting')
        if not isinstance(self.config['dependency_directed'], bool):
            raise ValueError('dependency_directed must be boolean')
        if self.config['initialization_importance'] != 'uniform':
            raise ValueError('Expected initialization_importance=uniform')
        warmup_steps = self.config['reconstruction_warmup_steps']
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError('reconstruction_warmup_steps must be a nonnegative integer')
        self.X = torch.as_tensor(X, dtype=torch.float32, device=device).detach().clone()
        if self.X.ndim != 2 or self.X.shape[1] < 1 or (not torch.isfinite(self.X).all()):
            raise ValueError('Invalid importance input features')
        self.graph = DependencyGraph.from_edges(len(self.X), edges, self.config['dependency_directed'])
        self.full_config = copy.deepcopy(cfg)
        self.node_order = list(range(len(self.X))) if node_order is None else list(node_order)
        if len(self.node_order) != len(self.X):
            raise ValueError('Node order mismatch')
        if role not in ('source', 'target'):
            raise ValueError('Expected source or target role')
        self.seed = int(cfg.get('seed', 321)) + 1000 * int(fold_id) + (100000 if role == 'target' else 0)
        self.metadata = dict(
            role=role,
            identifier=identifier,
            fold_id=int(fold_id),
            seed=self.seed,
            num_nodes=len(self.X),
            input_dim=self.X.shape[1],
            implementation_version='V2.1',
            dependency_directed=self.graph.directed,
            node_order=self.node_order,
            positive_pair_count=len(self.graph.positive_pairs)
        )
        if self.mode != 'paper_v2':
            self.metadata['importance_mode'] = self.mode
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.model = V2NodeImportanceModel(
                self.X.shape[1],
                self.config['hidden_dim'],
                self.config['dropout'],
                directed=self.graph.directed,
                score_bound_b=self.config['score_bound_b']
            ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.sampling_generator = torch.Generator().manual_seed(self.seed)
        self.current_step = 0
        (self.sync_history, self.step_history) = ([], [])
        self.previous_p = None
        from mvcha_dp.protocol.contract import is_parity
        self.parity = is_parity(cfg)
        self.fusion = None
        self.dependency_template = None
        if self.parity:
            from mvcha_dp.graph.fusion import DependencyTemplate
            self.dependency_template = DependencyTemplate(len(self.X), edges)
            self.metadata.update(
                model_version=cfg['model_version'],
                feature_order='TNC'
            )
        self.has_valid_ranking_update = False
        self.warmup_completed = False
        self.warmup_history = []

    def _loss(self, ranking_enabled=True):
        c = self.config
        return compute_v2_self_supervised_loss(
            self.model,
            self.X,
            self.graph,
            generator=self.sampling_generator,
            sample_size=c['num_candidates'],
            context_hops=c['context_hops'],
            rho=c['rho'],
            tau0=c['tau0'],
            negative_ratio=c['negative_ratio'],
            lambda_rec=c['lambda_rec'],
            lambda_imp=c['lambda_imp'],
            contribution_tie_margin=c['contribution_tie_margin'],
            max_positive_samples=c['max_positive_samples'],
            ranking_enabled=ranking_enabled,
            **{k: c[k] for k in (
                'context_pairs_per_class',
                'norm_std_floor_rec',
                'norm_std_floor_emb',
                'norm_clip',
                'zero_norm_eps'
            )}
        )

    def warmup(self):
        if self.warmup_completed:
            return
        count = self.config['reconstruction_warmup_steps'] if self.mode != 'frozen_random_importance' else 0
        for step in range(count):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            result = self._loss(ranking_enabled=False)
            if not torch.isfinite(result['loss_self']):
                raise FloatingPointError('Nonfinite reconstruction warmup loss')
            valid = result['diagnostics']['has_optimization_signal']
            if valid:
                result['loss_self'].backward()
                self.optimizer.step()
            self.warmup_history.append(dict(
                step=step,
                loss_rec=float(result['loss_rec'].detach()),
                optimizer_step=valid,
                reason=result['diagnostics']['reconstruction']['reason']
            ))
        self.optimizer.zero_grad(set_to_none=True)
        self.warmup_completed = True

    def compute_p(self):
        flags = [m.training for m in self.model.modules()]
        try:
            self.model.eval()
            with torch.no_grad():
                result = self.model(self.X, self.graph, self.config['tau0'])
                p = result['p'] if self.has_valid_ranking_update else torch.full_like(result['p'], 1.0 / len(self.X))
                return (result['bounded_score_s'].detach(), p.detach())
        finally:
            for (module, flag) in zip(self.model.modules(), flags):
                module.training = flag

    def self_supervised_step(self, epoch):
        if epoch != self.current_step:
            raise ValueError('Importance update epoch mismatch')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        c = self.config
        result = self._loss()
        if not torch.isfinite(result['loss_self']):
            raise FloatingPointError('Nonfinite importance loss')
        valid_update = result['diagnostics']['has_optimization_signal']
        if valid_update:
            result['loss_self'].backward()
        if any((p.grad is not None and (not torch.isfinite(p.grad).all()) for p in self.model.parameters())):
            raise FloatingPointError('Nonfinite importance gradient')
        head_before = parameter_bytes(self.model.head)
        ranking_gradient = any((p.grad is not None and bool((p.grad != 0).any()) for p in self.model.head.parameters()))
        if valid_update:
            self.optimizer.step()
        ranking_update = bool(valid_update and c['lambda_imp'] > 0 and (result['num_valid_pairs'] > 0) and ranking_gradient and (parameter_bytes(self.model.head) != head_before))
        self.has_valid_ranking_update |= ranking_update
        self.optimizer.zero_grad(set_to_none=True)
        self.current_step += 1
        record = dict(epoch=int(epoch), step=self.current_step, optimizer_step=valid_update,
                      ranking_update_occurred=ranking_update,
                      **{key: float(result[key].detach()) for key in ('loss_rec', 'loss_imp', 'loss_self')})
        self.step_history.append(record)
        return record


    def synchronize(self, epoch, views, edges, cfg, previous_H=None):
        self.warmup()
        if epoch != self.current_step or epoch % cfg.get('update_interval', 5):
            raise ValueError('Invalid hypergraph synchronization epoch')
        if self.sync_history and self.sync_history[-1]['epoch'] >= epoch:
            raise ValueError('Duplicate/out-of-order hypergraph synchronization')
        (scores, p) = self.compute_p()
        named = dict(zip(('TSM', 'CNM', 'NEM'), views))
        graph = build_mvcha_hypergraph(
            named if self.parity else views,
            self.X,
            edges,
            cfg,
            self.X.device,
            precomputed_importance=p.detach(),
            dependency_template=self.dependency_template
        )
        self.fusion = graph.get('fusion')
        H = torch.as_tensor(graph['H'], dtype=torch.float32, device=self.X.device)
        W = torch.as_tensor(graph['W'], dtype=torch.float32, device=self.X.device)
        self.sync_history.append(dict(epoch=int(epoch), p=p.detach().cpu().tolist()))
        return (H, W)

    def get_sampling_state(self):
        return self.sampling_generator.get_state().clone()



def create_runtime(X, view_data, cfg, device, role='source'):
    if not uses_v2(cfg):
        return None
    if not cfg.get('enable_periodic_update', False) or cfg.get('update_interval', 5) != 5:
        raise ValueError('Expected periodic graph updates every 5 epochs')
    if not cfg.get('use_node_importance', True):
        raise ValueError('Node importance is disabled')
    if view_data is None:
        raise ValueError('Missing view features or dependency edges')
    return V2ImportanceRuntime(
        X,
        view_data['E'],
        cfg,
        device,
        role,
        view_data.get('project', ''),
        cfg.get('fold_id', 0),
        view_data.get('node_order')
    )


def begin_importance_epoch(runtime, epoch, H, W, view_data, cfg):
    if epoch % cfg.get('update_interval', 5) == 0:
        views = [view_data.get(k) for k in ('X_origin', 'X_metric', 'X_vector')]
        (H, W) = runtime.synchronize(epoch, views, view_data['E'], cfg, H)
    record = runtime.self_supervised_step(epoch)
    return (H, W, record)
