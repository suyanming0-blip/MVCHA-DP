import numpy as np
from dhg import Graph, Hypergraph


class DependencyTemplate:

    def __init__(self, num_nodes, raw_edges):
        self.raw_edges = [tuple(map(int, e)) for e in raw_edges]
        self.template = Hypergraph.from_graph(Graph(num_nodes, self.raw_edges, extra_selfloop=False))
        (self.edges, self.weights) = self.template.e
    def fuse(self, view_names, view_edges):
        hg = self.template.clone()
        proposals = [dict(origin='dependency', edge_id=i, nodes=list(e)) for (i, e) in enumerate(self.edges)]
        for (name, edges) in zip(view_names, view_edges):
            if edges:
                hg.add_hyperedges([list(e) for e in edges], merge_op='mean', group_name='main')
            proposals.extend((dict(origin=name, center=i, nodes=list(e)) for (i, e) in enumerate(edges)))
        (edges, weights) = hg.e
        keys = [tuple(e) for e in edges]
        index = {key: i for (i, key) in enumerate(keys)}
        origins = [[] for _ in keys]
        for proposal in proposals:
            key = tuple(sorted(proposal['nodes']))
            proposal['merged_edge_id'] = index[key]
            origins[index[key]].append({k: v for (k, v) in proposal.items() if k != 'merged_edge_id'})
        assert all((w == 1 for w in weights))
        info = dict(
            merged_edge_keys=[list(k) for k in keys],
            origins=origins,
            raw_proposals=proposals
        )
        return (hg.H.to_dense().cpu().numpy().astype(np.float32), np.asarray(weights, dtype=np.float32), info)
