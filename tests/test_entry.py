import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import main as entry
from mvcha_dp.protocol.repeated import execute_plan


class EntryTests(unittest.TestCase):
    def test_no_arguments_runs_all_tasks(self):
        p = entry.parser()
        args = p.parse_args([])
        data = Path(entry.__file__).resolve().parent / 'data'
        with patch('pathlib.Path.mkdir'), \
             patch('mvcha_dp.protocol.repeated.prepare', return_value={'status': 'READY'}) as prepare, \
             patch('mvcha_dp.protocol.repeated.execute_plan', return_value=dict(status='COMPLETE', completed_runs=12080, expected_runs=12080)) as execute, \
             patch('main.run_one') as single:
            entry.dispatch(args, p)
        self.assertEqual(args.output.parent, Path(entry.__file__).resolve().parent / 'results')
        self.assertTrue(args.output.name.startswith('full_'))
        prepare.assert_called_once_with(data, args.output / 'plan', None, None)
        execute.assert_called_once_with(args.output / 'plan', data, args.output / 'results')
        single.assert_not_called()

    def test_old_options_rejected_before_dispatch(self):
        options = [['--data-package', 'data'], ['--output', 'out'], ['--scenario', 'WPDP'],
                   ['--task-id', 'task'], ['--list-tasks'], ['--prepare-plan'], ['--run-plan'],
                   ['--summarize'], ['--plan', 'plan'], ['--_run-id', 'x'],
                   ['--profile', 'default'], ['--run-id', 'x']]
        for args in options:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), \
                 patch('main.dispatch') as dispatch, self.assertRaises(SystemExit) as error:
                entry.main(args)
            self.assertEqual(error.exception.code, 2)
            dispatch.assert_not_called()

    def test_public_help(self):
        text = entry.parser().format_help()
        for option in ('--_worker', '--scenario', '--data-package', '--output', '--task-id'):
            self.assertNotIn(option, text)

    def test_blocked_plan_never_trains(self):
        p = entry.parser()
        with patch('pathlib.Path.mkdir'), \
             patch('mvcha_dp.protocol.repeated.prepare', return_value={'status': 'BLOCKED_INVALID_SPLIT'}), \
             patch('mvcha_dp.protocol.repeated.execute_plan') as execute, self.assertRaises(SystemExit):
            entry.dispatch(p.parse_args([]), p)
        execute.assert_not_called()

    def test_internal_worker_uses_verified_context(self):
        p = entry.parser()
        args = p.parse_args(['--_worker', 'plan', 'run', 'data', 'out'])
        context = dict(task_id='task', seed=322)
        with patch('mvcha_dp.protocol.repeated.resolve_run', return_value=context) as resolve, patch('main.run_one') as worker:
            entry.dispatch(args, p)
        resolve.assert_called_once_with(Path('plan'), Path('data'), 'run')
        worker.assert_called_once_with(args, context)
        self.assertEqual(args.task_id, 'task')
        self.assertEqual(args.output, Path('out'))

    def test_invalid_context_never_trains(self):
        p = entry.parser()
        args = p.parse_args(['--_worker', 'plan', 'run', 'data', 'out'])
        with patch('mvcha_dp.protocol.repeated.resolve_run', side_effect=ValueError('invalid plan')), \
             patch('main.run_one') as worker, self.assertRaises(ValueError):
            entry.dispatch(args, p)
        worker.assert_not_called()

    def test_dispatcher_worker_command_roundtrip(self):
        plan = dict(runs=[dict(run_id='t000_s321_f0')], expected_runs=1)
        with patch('mvcha_dp.protocol.repeated.load_plan', return_value=plan), \
             patch('pathlib.Path.mkdir'), patch('pathlib.Path.open', mock_open()), \
             patch('mvcha_dp.protocol.repeated.write'), \
             patch('mvcha_dp.protocol.repeated.summarize', return_value={'status': 'COMPLETE'}), \
             patch('mvcha_dp.protocol.repeated.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run:
            execute_plan('plan with spaces', 'data with spaces', 'results')
        command = run.call_args[0][0]
        p = entry.parser()
        args = p.parse_args(command[3:])
        context = dict(task_id='task', seed=321)
        with patch('mvcha_dp.protocol.repeated.resolve_run', return_value=context) as resolve, patch('main.run_one') as worker:
            entry.dispatch(args, p)
        resolve.assert_called_once_with(Path('plan with spaces').resolve(), Path('data with spaces').resolve(), 't000_s321_f0')
        self.assertEqual(args.output, Path('results/runs/t000_s321_f0').resolve())
        worker.assert_called_once_with(args, context)


if __name__ == '__main__':
    unittest.main()
