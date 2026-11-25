# Repeated-seed experiment execution and aggregation.

import copy
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .checks import require

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def recipe():
    value = read(ROOT / 'protocol/repeated20.json')
    require(value['seeds'] == list(range(321, 341)) and value['wpdp_folds'] == 5, 'Invalid seed list or fold count')
    return value


def make_wpdp_splits(task, original, labels, seed):
    y = np.asarray(labels)
    n = task['source_node_count']
    require(y.shape == (n,) and np.isin(y, [0, 1]).all(), 'Invalid stratification labels')
    counts = np.bincount(y.astype(np.int64), minlength=2)
    require(int(counts.min()) >= 5, 'MINORITY_COUNT_BELOW_5')
    folds = [test for _, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(n), y)]
    output = []
    for i in range(5):
        val = (i + 1) % 5
        train = np.sort(np.concatenate([folds[j] for j in range(5) if j not in (i, val)]))
        split = copy.deepcopy(original)
        split.update(seed=seed, split_seed=seed, fold=i, profile='repeated20', policy='stratified_five_fold_3_1_1',
                     ids=dict(train=train.tolist(), validation=folds[val].tolist(), evaluation=folds[i].tolist()))
        for ids in split['ids'].values():
            require(set(np.unique(y[ids])) == {0, 1}, 'SPLIT_CLASS_INVALID')
        output.append(split)
    return output


def make_cross_split(original, seed):
    split = copy.deepcopy(original)
    split.update(seed=seed, split_seed=original['seed'], fold=None, profile='repeated20',
                 policy='original_frozen_source_split_repeated_training')
    return split


def stratification_labels(package, task):
    from mvcha_dp.data.package import NumericPackage
    p = NumericPackage(package)
    item = task['source']
    entry = p.index[item['family'] + '/' + item['project']]
    folder = p.path / entry['folder']
    domain = read(folder / 'DOMAIN.json')
    label = domain['arrays']['labels']
    y = np.load(folder / label['file'], allow_pickle=False)
    require(list(y.shape) == label['shape'] and str(y.dtype) == label['dtype'], 'Label schema changed')
    return y


def prepare(package, output, task_id=None, scenario=None):
    from mvcha_dp.training.runner import load_task
    package, output = Path(package).resolve(), Path(output).resolve()
    policy = recipe()
    manifest = read(package / 'FINAL_FULL_TASK_MANIFEST.json')
    all_splits = {s['task_id']: s for s in read(package / 'FINAL_FULL_SPLITS.json')['splits']}
    tasks = [t for t in manifest['tasks'] if (task_id is None or t['task_id'] == task_id) and (scenario is None or t['scenario'] == scenario)]
    require(bool(tasks), 'No matching tasks')
    load_task(package, tasks[0]['task_id'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'splits').mkdir()
    plan = dict(schema='MVCHA_REPEATED20_PLAN_V1', status='READY', profile='repeated20', recipe=policy,
                tasks=tasks, runs=[],
                blockers=[], stratification_label_access=[], base_task_count=len(tasks))
    for index, task in enumerate(tasks):
        original = all_splits[task['task_id']]
        rows = {}
        labels = None
        try:
            if task['scenario'] == 'WPDP':
                labels = stratification_labels(package, task)
                counts = np.bincount(labels.astype(np.int64), minlength=2).tolist()
                plan['stratification_label_access'].append(dict(
                    task_id=task['task_id'],
                    purpose='partition_stratification_only',
                    class_counts=counts
                ))
                require(min(counts) >= 5, 'MINORITY_COUNT_BELOW_5')
            for seed in policy['seeds']:
                generated = make_wpdp_splits(task, original, labels, seed) if task['scenario'] == 'WPDP' else [make_cross_split(original, seed)]
                for split in generated:
                    run_id = f't{index:03d}_s{seed}_f{split["fold"] if split["fold"] is not None else "none"}'
                    rows[run_id] = split
                    plan['runs'].append(dict(run_id=run_id, task_id=task['task_id'], scenario=task['scenario'], seed=seed,
                                             fold=split['fold'], split_file=f'splits/t{index:03d}.json'))
        except ValueError as e:
            plan['blockers'].append(dict(task_id=task['task_id'], reason=str(e)))
        path = f'splits/t{index:03d}.json'
        write(output / path, rows)
    expected = sum((100 if t['scenario'] == 'WPDP' else 20) for t in tasks)
    plan['expected_runs'] = expected
    if plan['blockers'] or len(plan['runs']) != expected:
        plan['status'] = 'BLOCKED_INVALID_SPLIT'
    write(output / 'PLAN.json', plan)
    print(json.dumps(
        dict(
            status=plan['status'],
            tasks=len(tasks),
            runs=len(plan['runs']),
            expected_runs=expected,
            blockers=plan['blockers']
        ),
        indent=2
    ))
    return plan


def load_plan(directory, package):
    directory = Path(directory).resolve()
    plan = read(directory / 'PLAN.json')
    require(
        plan['schema'] == 'MVCHA_REPEATED20_PLAN_V1' and plan['status'] == 'READY',
        'Plan is not ready'
    )
    require(len(plan['runs']) == plan['expected_runs'], 'Incomplete run inventory')
    require(len({r['run_id'] for r in plan['runs']}) == len(plan['runs']), 'Duplicate runs')
    return plan


def resolve_run(directory, package, run_id):
    directory = Path(directory).resolve()
    plan = load_plan(directory, package)
    run = next((r for r in plan['runs'] if r['run_id'] == run_id), None)
    require(run is not None, 'Unknown repeated run ID')
    path = (directory / run['split_file']).resolve()
    require(path.is_relative_to(directory), 'Split path outside plan')
    split = read(path)[run_id]
    require(
        split['seed'] == run['seed'] and split['fold'] == run['fold'] and split['task_id'] == run['task_id'],
        'Run/split mismatch'
    )
    return dict(run, split=split, profile='repeated20')


def execute_plan(directory, package, output):
    plan = load_plan(directory, package)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    failures = []
    write(
        output / 'execution.json',
        dict(expected_runs=plan['expected_runs'], profile='repeated20')
    )
    for run in plan['runs']:
        rid = run['run_id']
        command = [sys.executable, '-B', str(ROOT / 'main.py'), '--_worker', str(Path(directory).resolve()),
                   rid, str(Path(package).resolve()), str(output / 'runs' / rid)]
        with (output / 'logs' / (rid + '.stdout.log')).open('x', encoding='utf8') as stdout, (output / 'logs' / (rid + '.stderr.log')).open('x', encoding='utf8') as stderr:
            code = subprocess.run(command, stdout=stdout, stderr=stderr).returncode
        if code:
            failures.append(dict(run_id=rid, exit_code=code))
        with (output / 'progress.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(dict(run_id=rid, exit_code=code)) + '\n')
        print(rid, 'COMPLETE' if code == 0 else 'FAILED', flush=True)
    write(output / 'failures.json', failures)
    return summarize(directory, package, output)


def distribution(values):
    from scipy.stats import t
    x = np.asarray(values, dtype=float)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = float(t.ppf(.975, len(x) - 1) * sd / np.sqrt(len(x)))
    q1, median, q3 = np.quantile(x, [.25, .5, .75])
    return dict(
        mean=mean,
        std=sd,
        ci95=[mean-half, mean+half],
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3-q1)
    )


def summarize(directory, package, output):
    plan = load_plan(directory, package)
    output = Path(output).resolve()
    grouped = defaultdict(list)
    incomplete = []
    for run in plan['runs']:
        path = output / 'runs' / run['run_id'] / 'result.json'
        if not path.exists():
            incomplete.append(run['run_id'])
            continue
        r = read(path)
        require(r['status'] == 'COMPLETE' and r['task_id'] == run['task_id'], 'Wrong result')
        require(
            r['repetition'] == dict(
                profile='repeated20',
                run_id=run['run_id'],
                seed=run['seed'],
                fold=run['fold']
            ),
            'Result identity differs from plan'
        )
        metrics = r['metrics']['final']
        require(
            all(metrics[k] is not None and np.isfinite(metrics[k]) for k in ('auc', 'f1', 'mcc')),
            'Missing or nonfinite result metric'
        )
        grouped[(run['scenario'], run['task_id'], run['seed'])].append(metrics)
    summary = dict(status='INCOMPLETE' if incomplete else 'COMPLETE', expected_runs=len(plan['runs']), completed_runs=len(plan['runs'])-len(incomplete), missing_runs=incomplete,
                   task_seed_means=[], scenario_seed_means={}, scenario_summary=None)
    if not incomplete:
        by_scenario_seed = defaultdict(list)
        for (scenario, task, seed), rows in grouped.items():
            require(len(rows) == (5 if scenario == 'WPDP' else 1), 'Wrong fold count')
            mean = {k: float(np.mean([r[k] for r in rows])) for k in ('auc', 'f1', 'mcc')}
            summary['task_seed_means'].append(dict(scenario=scenario, task_id=task, seed=seed, metrics=mean))
            by_scenario_seed[scenario, seed].append(mean)
        task_counts = Counter(t['scenario'] for t in plan['tasks'])
        for scenario, count in task_counts.items():
            seed_means = []
            for seed in plan['recipe']['seeds']:
                rows = by_scenario_seed[scenario, seed]
                require(len(rows) == count, 'Missing task in scenario mean')
                seed_means.append(dict(
                    seed=seed,
                    **{k: float(np.mean([r[k] for r in rows])) for k in ('auc', 'f1', 'mcc')}
                ))
            summary['scenario_seed_means'][scenario] = seed_means
        summary['scenario_summary'] = {
            scenario: {k: distribution([r[k] for r in rows]) for k in ('auc', 'f1', 'mcc')} for scenario,
            rows in summary['scenario_seed_means'].items()
        }
    write(output / 'summary.json', summary)
    return summary
