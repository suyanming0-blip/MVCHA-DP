import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mvcha_dp.protocol.selection import Scope, Validation, selected_checkpoint
from mvcha_dp.training.snapshots import Snapshots
from mvcha_dp.training.candidate_cache import CandidateCache


class SnapshotTests(unittest.TestCase):
    def test_graph_pairing_and_historical_reselection(self):
        for budget in (0, 1024 * 1024):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as folder:
                store = SimpleNamespace(folder=lambda task: Path(folder))
                with patch('mvcha_dp.training.candidate_cache.CandidateCache', side_effect=lambda: CandidateCache(budget)):
                    snapshots = Snapshots(store, 'task')
                scope = Scope('WPDP', 'Promise', 'task', 'inner_validation', seed=322)
                history, originals = [], {}
                for epoch in range(7):
                    auc = .8 if epoch < 5 else .802 if epoch == 5 else .8021
                    f1, mcc = (.8, .7) if epoch == 0 else (.5, .4)
                    if epoch == 6:
                        f1, mcc = .4, .3
                    history.append(Validation.create(scope, epoch, auc=auc, f1=f1, mcc=mcc))
                    sync = 0 if epoch < 5 else 5
                    state = dict(epoch=epoch, model={'weight': torch.tensor([float(epoch)])},
                                 graphs={'source': (torch.full((2, 2), float(sync + 1)), torch.ones(2))},
                                 importance={'source': {'active_sync': {'epoch': sync}}})
                    originals[epoch] = state
                    snapshots.snapshot(state, history)
                    if epoch == 5:
                        self.assertEqual(selected_checkpoint(history).epoch, 0)
                        restored = snapshots.restore(0)
                        self.assertTrue(torch.equal(restored['graphs']['source'][0], originals[0]['graphs']['source'][0]))
                chosen = selected_checkpoint(history).epoch
                self.assertEqual(chosen, 5)
                restored = snapshots.restore(chosen)
                self.assertTrue(torch.equal(restored['model']['weight'], originals[chosen]['model']['weight']))
                self.assertTrue(torch.equal(restored['graphs']['source'][0], originals[chosen]['graphs']['source'][0]))
                self.assertEqual(restored['importance']['source']['active_sync']['epoch'], 5)


if __name__ == '__main__':
    unittest.main()
