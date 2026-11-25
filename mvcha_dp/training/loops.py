# Neural training loops.

import copy
import numpy as np
import torch
from sklearn.metrics import f1_score
from mvcha_dp.models.hgnn import HGNN
from mvcha_dp.protocol.contract import validate
from mvcha_dp.evaluation.logistic import SplitClassInvalid
from mvcha_dp.training.randomness import set_seed
from mvcha_dp.training.importance_runtime import create_runtime, begin_importance_epoch
from mvcha_dp.training.state import capture_checkpoint


def guard(cfg, device):
    validate(cfg)
    if torch.device(device).type != 'cuda' and (not cfg.get('mechanism_test_only', False)):
        raise RuntimeError('CUDA is required')
    if cfg.get('importance_mode') != 'paper_v2':
        raise ValueError('Expected importance_mode=paper_v2')


def encoder(model):
    return model.hgnn if hasattr(model, 'hgnn') else model


def attach_metadata(model, cfg, indices):
    model.cfg = copy.deepcopy(cfg)
    model.split_ids = {k: list(map(int, v)) for (k, v) in indices.items()}
    model.stage = 'neural_training'


def train_wpdp(X, y_train, train_index, y_val, val_index, cfg, device, view_data, *, on_epoch):
    guard(cfg, device)
    set_seed(cfg['seed'])
    if len(np.unique(y_train)) != 2:
        raise SplitClassInvalid('SPLIT_CLASS_INVALID: WPDP train')
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    tr = torch.as_tensor(train_index, dtype=torch.long, device=device)
    va = torch.as_tensor(val_index, dtype=torch.long, device=device)
    if set(tr.tolist()) & set(va.tolist()):
        raise ValueError('Train/validation overlap')
    yt = torch.as_tensor(y_train, dtype=torch.long, device=device)
    runtime = create_runtime(X, view_data, cfg, device)
    model = HGNN(X.shape[1], cfg['n_hid'], 0.5, profile='WPDP').to(device)
    attach_metadata(
        model,
        cfg,
        {
            'train': train_index,
            'validation': val_index,
            'evaluation': sorted(set(range(len(X))) - set(train_index) - set(val_index))
        }
    )
    model.importance_runtimes = {'source': runtime}
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, cfg['milestones'], cfg['gamma'])
    history = []
    start = 0
    H = W = None
    for epoch in range(start, cfg['max_epoch']):
        (H, W, _) = begin_importance_epoch(runtime, epoch, H, W, view_data, cfg)
        model.train()
        opt.zero_grad(set_to_none=True)
        (logits, _) = model(X, H, W)
        loss = torch.nn.functional.cross_entropy(logits[tr], yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            (vl, _) = model(X, H, W)
        vf = float(f1_score(y_val, vl[va].argmax(1).cpu().numpy(), zero_division=0))
        history.append(dict(
            epoch=epoch,
            real_loss=float(loss.detach()),
            val_f1=vf,
            refreshed=epoch % 5 == 0,
            selection_used=False
        ))
        scheduler.step()
        model.stage = 'neural_complete' if epoch == cfg['max_epoch'] - 1 else 'neural_training'
        cp = on_epoch(model, capture_checkpoint(model, epoch, vf, source=(H, W)), {'main': opt}, [scheduler], history)
        model.best_checkpoint = cp
        model.last_checkpoint = cp
        print(f'WPDP parity epoch={epoch} CE={float(loss.detach()):.6f} validation_diagnostic={vf:.4f}', flush=True)
    raise RuntimeError('ES controller did not terminate at the configured boundary')


def train_cross(model, X_s, y_s, train_idx, val_idx, X_t, view_s, view_t, cfg, device, *, on_epoch):
    from mvcha_dp.models.cross_domain import adaptation_step
    guard(cfg, device)
    y_s = y_s.detach().cpu().numpy() if torch.is_tensor(y_s) else np.asarray(y_s)
    if len(np.unique(np.asarray(y_s)[train_idx])) != 2:
        raise SplitClassInvalid('SPLIT_CLASS_INVALID: source train')
    model.to(device)
    (X_s, X_t) = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in (X_s, X_t)]
    ys = torch.as_tensor(y_s, dtype=torch.long, device=device)
    (tr, va) = [torch.as_tensor(i, dtype=torch.long, device=device) for i in (train_idx, val_idx)]
    if set(tr.tolist()) & set(va.tolist()):
        raise ValueError('Source train/validation overlap')
    attach_metadata(model, cfg, {'train': train_idx, 'validation': val_idx, 'target': range(len(X_t))})
    runtimes = {role: create_runtime(x, v, cfg, device, role) for (role, x, v) in [
        ('source', X_s, view_s),
        ('target', X_t, view_t)
    ]}
    model.importance_runtimes = runtimes
    for layer in (model.hgnn.linear1, model.hgnn.linear2):
        layer.requires_grad_(False)
    kwargs = dict(lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    opts = {
        'encoder': torch.optim.Adam(model.hgnn.encoder_parameters(), **kwargs),
        'classifier': torch.optim.Adam(model.classifier.parameters(), **kwargs),
        'domain': torch.optim.Adam(model.domain_disc.parameters(), **kwargs)
    }
    sched = [torch.optim.lr_scheduler.MultiStepLR(o, cfg['milestones'], cfg['gamma']) for o in opts.values()]
    history = []
    start = 0
    graphs = {'source': (None, None), 'target': (None, None)}
    for epoch in range(start, cfg['max_epoch']):
        if epoch % 5 == 0:
            for (role, v) in [('source', view_s), ('target', view_t)]:
                rt = runtimes[role]
                graphs[role] = rt.synchronize(
                    epoch,
                    [v.get(k) for k in ('X_origin', 'X_metric', 'X_vector')],
                    v['E'],
                    cfg,
                    graphs[role][0]
                )
        for role in graphs:
            runtimes[role].self_supervised_step(epoch)
        (hs, ws) = graphs['source']
        (ht, wt) = graphs['target']
        args = (X_s, ys, tr, hs, ws, X_t, ht, wt, cfg)
        adaptation_step(model, opts['domain'], 'domain', *args)
        for _ in range(2):
            adaptation_step(model, opts['classifier'], 'classifier', *args)
        losses = adaptation_step(model, opts['encoder'], 'encoder', *args)
        model.eval()
        with torch.no_grad():
            (_, a, b, _) = model(X_s, hs, ws)
        prob = (a[va].softmax(1)[:, 1] + b[va].softmax(1)[:, 1]) / 2
        vf = float(f1_score(ys[va].cpu().numpy(), (prob >= 0.5).cpu().numpy(), zero_division=0))
        history.append(dict(epoch=epoch, val_f1=vf, selection_used=False, refreshed=epoch % 5 == 0, **losses))
        for s in sched:
            s.step()
        model.stage = 'neural_complete' if epoch == cfg['max_epoch'] - 1 else 'neural_training'
        cp = on_epoch(model, capture_checkpoint(model, epoch, vf, **graphs), opts, sched, history)
        model.best_checkpoint = cp
        model.last_checkpoint = cp
    raise RuntimeError('ES controller did not terminate at the configured boundary')
