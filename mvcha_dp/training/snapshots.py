from mvcha_dp.protocol.checks import require
from mvcha_dp.protocol.selection import validate_history, selected_checkpoint, AUC_TOLERANCE
import os
import torch
SCHEMA = 'MVCHA_SELECTED_EPOCH_EVAL_SNAPSHOT_V1'


class Snapshots:

    def __init__(self, store, tid):
        self.store = store
        self.tid = tid
        self.path = store.folder(tid) / 'eval_snapshots'
        require(not self.path.exists(), 'Snapshot directory already exists')
        self.path.mkdir()
        self.saved = []
        self.pruned = []
        self.backend = 'compact'
        require(self.backend in ('disk', 'compact'), 'Unknown candidate storage backend')
        self.cache = None
        if self.backend == 'compact':
            from .candidate_cache import CandidateCache
            self.cache = CandidateCache()
        self.storage_stats = dict(backend=self.backend, durable_writes=0, durable_bytes=0)

    def file(self, epoch):
        require(type(epoch) is int and 0 <= epoch < 200, 'Invalid epoch')
        p = self.path / f'epoch_{epoch:03d}.pt'
        require(not p.is_symlink() and p.resolve().parent == self.path.resolve(), 'Invalid snapshot path')
        return p

    def snapshot(self, full, history):
        validate_history(history)
        epoch = full['epoch']
        require(epoch == len(history) - 1 and epoch == len(self.saved), 'Metric/state epoch mismatch')
        peak = max((r.auc for r in history))
        eligible = {r.epoch for r in history if r.auc >= peak - AUC_TOLERANCE}
        require(selected_checkpoint(history).epoch in eligible, 'Selector/retention mismatch')
        if self.cache is not None:
            self.cache.prune(eligible)
            self._prune_files(eligible)
            if epoch in eligible:
                state = full
                if not self.cache.retain(state):
                    self._write(state)
            else:
                self.cache.stats['skipped_ineligible'] += 1
            self.saved.append(epoch)
            chosen = selected_checkpoint(history).epoch
            require(chosen in self.cache.candidates or self.file(chosen).is_file(), 'Selected candidate missing')
            return
        state = full
        self._write(state)
        self.saved.append(epoch)
        self._prune_files(eligible)

    def _write(self, state):
        epoch = state['epoch']
        value = dict(
            schema=SCHEMA,
            epoch=epoch,
            state=state
        )
        path = self.file(epoch)
        require(not path.exists(), 'Snapshot file already exists')
        tmp = path.with_suffix('.tmp')
        try:
            with tmp.open('wb') as f:
                torch.save(value, f)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)
            self.storage_stats['durable_writes'] += 1
            self.storage_stats['durable_bytes'] += path.stat().st_size
        finally:
            if tmp.exists():
                tmp.unlink()

    def _prune_files(self, eligible):
        for p in self.path.glob('epoch_*.pt'):
            i = int(p.stem.split('_')[1])
            if i not in eligible:
                require(p.resolve().parent == self.path.resolve() and (not p.is_symlink()), 'Invalid snapshot removal path')
                p.unlink()
                self.pruned.append(i)

    def restore(self, epoch):
        if self.cache is not None and epoch in self.cache.candidates:
            state = self.cache.get(epoch)
            if not self.file(epoch).exists():
                self._write(state)
        value = torch.load(self.file(epoch), map_location='cpu')
        require(
            set(value) == {'schema', 'epoch', 'state'} and value['schema'] == SCHEMA,
            'Unknown schema'
        )
        require(value['epoch'] == epoch, 'Snapshot epoch mismatch')
        return value['state']
