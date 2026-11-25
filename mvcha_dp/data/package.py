from pathlib import Path
from dataclasses import dataclass
from mvcha_dp.protocol.checks import require


@dataclass(frozen=True)
class Domain:
    X: object
    view_data: dict
    metadata: dict
    folder: Path


class NativeLabelAccess:

    def __init__(self, domain, role, split=None):
        require(role in ('source', 'target'), 'Unknown domain role')
        self.domain = domain
        self.role = role
        self.split = split
        self.stage = 'preflight'
        self.access_log = []

    def begin_training(self):
        require(self.stage == 'preflight', 'Invalid stage transition')
        self.stage = 'neural_training'

    def mark_lr_complete(self, model):
        require(
            self.stage == 'neural_training' and getattr(model, 'stage', None) == 'lr_complete' and (getattr(model, 'final_lr', None) is not None),
            'Final LR stage required'
        )
        self.stage = 'lr_complete'
import json
import numpy as np
LABEL_RECORDS = {}


class NumericPackage:

    def __init__(self, path):
        self.path = Path(path).resolve()
        manifest = self.path / 'PACKAGE.json'
        self.doc = json.loads(manifest.read_text(encoding='utf8'))
        require(self.doc['schema'] == 'MVCHA_NUMERIC_INPUT_V1', 'Unknown package schema')
        self.index = {x['domain']: x for x in self.doc['domains']}

    def load(self, item):
        entry = self.index[item['family'] + '/' + item['project']]
        folder = self.path / entry['folder']
        d = json.loads((folder / 'DOMAIN.json').read_text(encoding='utf8'))
        for (k, v) in item.items():
            require(d['upstream_metadata'][k] == v, 'Wrong domain ' + k)
        arrays = {}
        for (name, r) in d['arrays'].items():
            if name == 'labels':
                continue
            a = np.load(folder / r['file'], allow_pickle=False)
            require(list(a.shape) == r['shape'] and str(a.dtype) == r['dtype'], 'Array schema changed')
            arrays[name] = a
        views = {k: arrays[k] for k in ('X_origin', 'X_vector', 'X_metric')}
        views.update(
            E=[tuple(map(int, x)) for x in arrays['edges']],
            node_order=d['anonymous_node_order'],
            project=item['family'] + '/' + item['project']
        )
        sims = {k: arrays[k + '_structural_similarity'] for k in views if k.startswith('X_')}
        LABEL_RECORDS[str(folder.resolve())] = dict(d['arrays']['labels'])
        return (Domain(arrays['INPUT69'], views, d['upstream_metadata'], folder), sims)


class PackageLabels(NativeLabelAccess):

    def read(self, purpose):
        require(purpose in ('train', 'validation', 'evaluation'), 'Unknown label purpose')
        if purpose != 'evaluation':
            require(self.role == 'source' and self.stage == 'neural_training', 'Invalid label role or stage')
        else:
            require(self.stage == 'lr_complete', 'Evaluation labels require final LR completion')
        ids = list(range(self.domain.metadata['node_count'])) if self.role == 'target' else self.split[purpose]
        require(str(self.domain.folder.resolve()) in LABEL_RECORDS, 'Domain was not loaded')
        r = LABEL_RECORDS[str(self.domain.folder.resolve())]
        p = self.domain.folder / r['file']
        a = np.load(p, allow_pickle=False, mmap_mode='r')
        require(str(a.dtype) == r['dtype'] and list(a.shape) == r['shape'], 'Label shape/dtype changed')
        y = np.asarray(a[ids]).copy()
        require(np.isin(y, [0, 1]).all(), 'Nonbinary label')
        self.access_log.append(dict(role=self.role, purpose=purpose, count=len(ids)))
        return y
