import copy
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mvcha_dp.protocol.repeated import make_wpdp_splits, make_cross_split, distribution, recipe, summarize
from mvcha_dp.protocol.selection import Scope
from mvcha_dp.evaluation.readout import isolated_readout_rng


class RepeatedTests(unittest.TestCase):
    def setUp(self):
        self.task = dict(task_id='WPDP/Promise/example', source_node_count=103)
        self.original = dict(task_id=self.task['task_id'], seed=321,
                             roles=dict(train='inner_train', validation='inner_validation', evaluation='outer_test'),
                             ids=dict(train=[0], validation=[1], evaluation=[2]))
        self.y = np.array([0] * 79 + [1] * 24)

    def test_five_fold_rotation_and_reproducibility(self):
        before = copy.deepcopy(self.original)
        folds = make_wpdp_splits(self.task, self.original, self.y, 321)
        self.assertEqual(folds, make_wpdp_splits(self.task, self.original, self.y, 321))
        self.assertNotEqual(folds, make_wpdp_splits(self.task, self.original, self.y, 322))
        visits = {k: np.zeros(len(self.y), dtype=int) for k in self.original['ids']}
        for s in folds:
            tr, va, te = [set(s['ids'][k]) for k in ('train', 'validation', 'evaluation')]
            self.assertFalse(tr & va or tr & te or va & te)
            self.assertEqual(tr | va | te, set(range(len(self.y))))
            for k, ids in s['ids'].items():
                visits[k][ids] += 1
                self.assertEqual(set(self.y[ids]), {0, 1})
        self.assertTrue(np.all(visits['train'] == 3))
        self.assertTrue(np.all(visits['validation'] == 1))
        self.assertTrue(np.all(visits['evaluation'] == 1))
        self.assertEqual(before, self.original)

    def test_small_minority_blocks(self):
        with self.assertRaisesRegex(ValueError, 'MINORITY_COUNT_BELOW_5'):
            make_wpdp_splits(self.task, self.original, np.array([0] * 99 + [1] * 4), 321)

    def test_cross_split_fixed_for_all_seeds(self):
        for seed in recipe()['seeds']:
            s = make_cross_split(self.original, seed)
            self.assertEqual(s['ids'], self.original['ids'])
            self.assertEqual(s['seed'], seed)
            self.assertEqual(s['split_seed'], 321)

    def test_scope_seed_range(self):
        values = ('WPDP', 'Promise', 'x', 'inner_validation')
        for seed in recipe()['seeds']:
            Scope(*values, seed=seed).validate()
        with self.assertRaises(ValueError):
            Scope(*values, seed=341).validate()

    def test_readout_rng_restores_streams(self):
        with patch('torch.cuda.is_initialized', return_value=False):
            random.seed(876)
            np.random.seed(876)
            torch.random.default_generator.manual_seed(876)
            before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
            with isolated_readout_rng(322):
                a = np.random.rand(5)
                random.random()
                torch.rand(2)
            self.assertEqual(random.getstate(), before[0])
            self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
            with isolated_readout_rng(322):
                self.assertTrue(np.array_equal(a, np.random.rand(5)))
            with isolated_readout_rng(323):
                self.assertFalse(np.array_equal(a, np.random.rand(5)))

    def test_sample_standard_deviation(self):
        s = distribution(list(range(20)))
        self.assertEqual(s['mean'], 9.5)
        self.assertEqual(s['std'], float(np.std(range(20), ddof=1)))
        self.assertLess(s['ci95'][0], s['mean'])
        self.assertGreater(s['ci95'][1], s['mean'])

    def test_aggregation_uses_all_folds_and_all_seeds(self):
        root = Path('synthetic_aggregation').resolve()
        plan_dir, output = root / 'plan', root / 'complete'
        runs = [dict(run_id=f'{seed}_{fold}', task_id='example', scenario='WPDP', seed=seed, fold=fold)
                for seed in recipe()['seeds'] for fold in range(5)]
        plan = dict(tasks=[dict(scenario='WPDP')], runs=runs, recipe=recipe())
        files = {}
        for run in runs:
            value = .5 + .01 * run['fold']
            files[output / 'runs' / run['run_id'] / 'result.json'] = dict(status='COMPLETE', task_id='example',
                repetition=dict(
                    profile='repeated20',
                    run_id=run['run_id'],
                    seed=run['seed'],
                    fold=run['fold'],
                ),
                metrics=dict(final=dict(auc=value, f1=value, mcc=value)))
        with patch('mvcha_dp.protocol.repeated.load_plan', return_value=plan), \
             patch('pathlib.Path.exists', lambda p: p in files), \
             patch('mvcha_dp.protocol.repeated.read', side_effect=lambda p: files[p]), \
             patch('mvcha_dp.protocol.repeated.write'):
            result = summarize(plan_dir, root, output)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(len(result['task_seed_means']), 20)
            self.assertAlmostEqual(result['scenario_summary']['WPDP']['auc']['mean'], .52)
            missing = summarize(plan_dir, root, root / 'partial')
            self.assertEqual(missing['status'], 'INCOMPLETE')
            self.assertIsNone(missing['scenario_summary'])


if __name__ == '__main__':
    unittest.main()
