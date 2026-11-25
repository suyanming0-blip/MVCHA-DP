# Per-task training and evaluation.

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mvcha_dp.data.package import NumericPackage, PackageLabels
from mvcha_dp.data.similarities import register, clear
from mvcha_dp.evaluation.metrics import evaluate
from mvcha_dp.evaluation.readout import lr_validation, refit_selected, extract_latent
from mvcha_dp.models.hgnn import HGNN
from mvcha_dp.models.cross_domain import MVCHADP
from mvcha_dp.protocol.checks import require
from mvcha_dp.protocol.selection import Scope, EarlyStopping
from mvcha_dp.training import loops
from mvcha_dp.training.randomness import set_seed
from mvcha_dp.training.snapshots import Snapshots


class BoundaryStop(Exception):
    pass


def load_task(package_path, task_id):
    package_path = Path(package_path)
    manifest = json.loads((package_path / 'FINAL_FULL_TASK_MANIFEST.json').read_text(encoding='utf8'))
    splits = json.loads((package_path / 'FINAL_FULL_SPLITS.json').read_text(encoding='utf8'))
    task = next((t for t in manifest['tasks'] if t['task_id'] == task_id), None)
    split = next((s for s in splits['splits'] if s['task_id'] == task_id), None)
    require(task is not None and split is not None, 'Unknown task ID')
    return task, split



class Trainer:
    def __init__(self, package_path, task_id, output, *, repeat_context=None):
        require(torch.cuda.is_available(), 'CUDA is required')
        require(
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'Expected CUBLAS_WORKSPACE_CONFIG=:4096:8'
        )
        torch.use_deterministic_algorithms(True)
        self.seed = 321 if repeat_context is None else repeat_context['seed']
        require(
            type(self.seed) is int and (self.seed == 321 if repeat_context is None else self.seed in range(321, 341)),
            'Invalid run seed'
        )
        set_seed(self.seed)
        self.task, self.split = load_task(package_path, task_id)
        if repeat_context is not None:
            require(
                repeat_context['task_id'] == task_id and repeat_context['profile'] == 'repeated20',
                'Wrong repeated run context'
            )
            self.split = repeat_context['split']
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        cfg_path = Path(__file__).resolve().parents[2] / 'configs' / (self.task['scenario'] + '__' + self.task['dataset_family'] + '.json')
        self.cfg = json.loads(cfg_path.read_text(encoding='utf8'))
        if repeat_context is not None:
            self.cfg['seed'] = self.seed
        pack = NumericPackage(package_path)
        clear()
        self.source, sims = pack.load(self.task['source'])
        register(self.source.view_data, sims)
        self.target = None
        if self.task['target']:
            self.target, sims = pack.load(self.task['target'])
            register(self.target.view_data, sims)
        self.labels = PackageLabels(self.source, 'source', self.split['ids'])
        self.labels.begin_training()
        self.ytrain = self.labels.read('train')
        self.yval = self.labels.read('validation')
        require(set(np.unique(self.ytrain)) == {0, 1} and set(np.unique(self.yval)) == {0, 1}, 'SPLIT_CLASS_INVALID')
        protocol = 'MVCHA_DP_V2_2_REVIEW' if repeat_context is None else 'MVCHA_DP_REPEATED20'
        self.scope = Scope(protocol, self.task['scenario'], self.task['dataset_family'], task_id,
                           self.split['roles']['validation'], seed=self.seed)
        self.execution = dict(task_id=task_id, seed=self.seed)
        if repeat_context is not None:
            self.execution.update(profile='repeated20', run_id=repeat_context['run_id'], fold=repeat_context['fold'])


    def folder(self, task_id):
        require(task_id == self.task['task_id'], 'Wrong task snapshot')
        return self.output

    def args(self, model, cp):
        g = loops.encoder(model)
        device = next(g.parameters()).device
        x = torch.as_tensor(self.source.X, device=device)
        h, w = [v.to(device) for v in cp['graphs']['source']]
        ids = self.split['ids']

        def latent():
            z = extract_latent(g, lambda: g.encode(x, h, w))
            return z[ids['train']], z[ids['validation']]

        return dict(train_role=self.split['roles']['train'], validation_role=self.split['roles']['validation'],
                    train_ids=ids['train'], validation_ids=ids['validation'],
                    read_train_labels=lambda: self.ytrain.copy(), read_validation_labels=lambda: self.yval.copy(),
                    latent_provider=latent, h_sync_epoch=cp['importance']['source']['active_sync']['epoch'])

    def run(self, event=lambda **k: None):
        controller = EarlyStopping(self.scope)
        rows = []
        latest, selected_epoch, last_state = None, None, None
        snapshots = Snapshots(self, self.task['task_id'])

        def boundary(model, cp, optimizers, schedulers, history, stale=0):
            nonlocal latest, selected_epoch, last_state
            full = cp
            epoch = full['epoch']
            latest = full
            row, _, _ = lr_validation(self.scope, epoch, **self.args(model, full))
            rows.append(row)
            last_state = controller.append(row)
            selected_epoch = controller.selected_epoch
            snapshots.snapshot(full, rows)
            event(kind='snapshot_epoch', epoch=epoch, selected_epoch=selected_epoch, controller=last_state)
            if controller.stopped:
                raise BoundaryStop()
            return full

        set_seed(self.seed)
        ids, s, t = self.split['ids'], self.source, self.target
        try:
            if self.task['scenario'] == 'WPDP':
                loops.train_wpdp(s.X, self.ytrain, ids['train'], self.yval, ids['validation'], self.cfg,
                                 torch.device('cuda:0'), s.view_data, on_epoch=boundary)
            else:
                model = MVCHADP(69, self.cfg['n_hid'], self.cfg['drop_out'], self.cfg)
                ys = np.zeros(len(s.X), dtype=np.int64)
                ys[ids['train']] = self.ytrain
                ys[ids['validation']] = self.yval
                loops.train_cross(model, s.X, ys, ids['train'], ids['validation'], t.X, s.view_data, t.view_data,
                                  self.cfg, torch.device('cuda:0'), on_epoch=boundary)
        except BoundaryStop:
            pass
        require(latest is not None and controller.stopped, 'Incomplete training')
        selected = snapshots.restore(selected_epoch)
        return dict(checkpoint=latest, selected=selected, selected_epoch=selected_epoch, history=rows,
                    controller=last_state, execution_identity=self.execution)

    def final_evaluate(self, result):
        cp = result['selected']
        scenario = self.task['scenario']
        model = HGNN(
            69,
            self.cfg['n_hid'],
            self.cfg['drop_out'],
            profile='WPDP'
        ) if scenario == 'WPDP' else MVCHADP(69, self.cfg['n_hid'], self.cfg['drop_out'], self.cfg)
        model.load_state_dict(cp['model'])
        model.cuda()
        model.eval()
        row, meta, lr = refit_selected(result['history'][result['selected_epoch']], **self.args(model, cp))
        domain = self.source if scenario == 'WPDP' else self.target
        role = 'source' if scenario == 'WPDP' else 'target'
        labels = PackageLabels(domain, role, self.split['ids'] if scenario == 'WPDP' else None)
        labels.begin_training()
        labels.mark_lr_complete(SimpleNamespace(stage='lr_complete', final_lr=lr))
        ids = self.split['ids']['evaluation'] if scenario == 'WPDP' else list(range(len(domain.X)))
        h, w = [x.cuda() for x in cp['graphs'][role]]
        with torch.no_grad():
            z = loops.encoder(model).encode(torch.as_tensor(domain.X, device='cuda'), h, w)[ids].cpu().numpy()
        final = evaluate(lr, z, labels.read('evaluation'))
        self.final_lr = lr
        return dict(validation=row.as_dict(), final=final, resampling=meta,
                    label_access=self.labels.access_log + labels.access_log)
