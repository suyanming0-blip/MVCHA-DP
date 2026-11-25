import torch
from mvcha_dp.protocol.checks import require


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        require(value.device.type == 'cpu', 'Expected a CPU snapshot')
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum((tensor_bytes(v) for v in value.values()))
    if isinstance(value, (tuple, list)):
        return sum((tensor_bytes(v) for v in value))
    return 0


class CandidateCache:

    def __init__(self, budget_bytes=256 * 1024 * 1024):
        require(type(budget_bytes) is int and budget_bytes >= 0, 'Invalid candidate memory budget')
        self.budget = budget_bytes
        self.candidates = {}
        self.graphs = {}
        self.bytes = 0
        self.stats = dict(
            peak_tensor_bytes=0,
            memory_candidates=0,
            spilled_candidates=0,
            shared_graph_hits=0,
            skipped_ineligible=0
        )

    def prune(self, eligible):
        for epoch in list(self.candidates):
            if epoch not in eligible:
                value = self.candidates.pop(epoch)
                self.bytes -= value['bytes']
        used = {v['graph_key'] for v in self.candidates.values()}
        for key in list(self.graphs):
            if key not in used:
                self.bytes -= self.graphs.pop(key)['bytes']
        require(0 <= self.bytes <= self.budget, 'Candidate memory accounting mismatch')

    def retain(self, state):
        epoch = state['epoch']
        require(epoch not in self.candidates, 'Duplicate candidate')
        key = tuple((role, value['active_sync']['epoch']) for role, value in sorted(state['importance'].items()))
        model_bytes = tensor_bytes(state['model'])
        graph_bytes = 0 if key in self.graphs else tensor_bytes(state['graphs'])
        if self.bytes + model_bytes + graph_bytes > self.budget:
            self.stats['spilled_candidates'] += 1
            return False
        if key not in self.graphs:
            self.graphs[key] = dict(value=state['graphs'], bytes=graph_bytes)
        else:
            self.stats['shared_graph_hits'] += 1
        self.candidates[epoch] = dict(
            model=state['model'],
            importance=state['importance'],
            graph_key=key,
            bytes=model_bytes
        )
        self.bytes += model_bytes + graph_bytes
        self.stats['memory_candidates'] += 1
        self.stats['peak_tensor_bytes'] = max(self.stats['peak_tensor_bytes'], self.bytes)
        return True

    def get(self, epoch):
        value = self.candidates[epoch]
        return dict(
            epoch=epoch,
            model=value['model'],
            importance=value['importance'],
            graphs=self.graphs[value['graph_key']]['value']
        )
