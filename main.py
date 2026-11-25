import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def parser():
    p = argparse.ArgumentParser(description='MVCHA-DP: full 20-seed WPDP/CVDP/CPDP experiment', allow_abbrev=False)
    p.add_argument('--_worker', nargs=4, metavar=('PLAN', 'RUN_ID', 'DATA', 'OUTPUT'), help=argparse.SUPPRESS)
    return p


def dispatch(args, p):
    from mvcha_dp.protocol.repeated import prepare, resolve_run, execute_plan
    if args._worker is not None:
        plan, run_id, data, output = args._worker
        args.data_package = Path(data)
        args.output = Path(output)
        context = resolve_run(Path(plan), args.data_package, run_id)
        args.task_id = context['task_id']
        run_one(args, context)
        return
    root = Path(__file__).resolve().parent
    args.data_package = root / 'data'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    args.output = root / 'results' / ('full_' + stamp)
    args.output.mkdir(parents=True, exist_ok=False)
    plan_path = args.output / 'plan'
    prepared = prepare(args.data_package, plan_path, None, None)
    if prepared['status'] != 'READY':
        raise SystemExit(2)
    result = execute_plan(plan_path, args.data_package, args.output / 'results')
    print(json.dumps(dict(status=result['status'], completed_runs=result['completed_runs'], expected_runs=result['expected_runs'])))
    if result['status'] != 'COMPLETE':
        raise SystemExit(2)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.dont_write_bytecode = True
    dispatch(args, p)


def run_one(args, context):
    from mvcha_dp.training.runner import Trainer
    trainer = Trainer(args.data_package, args.task_id, args.output, repeat_context=context)
    write_json(trainer.output / 'configuration.json', dict(task=trainer.task, split=trainer.split, config=trainer.cfg,
                                                         run=trainer.execution))

    def event(**value):
        row = dict(epoch=value['epoch'], selected_epoch=value['selected_epoch'], controller=value['controller'])
        with (trainer.output / 'progress.jsonl').open('a', encoding='utf8') as handle:
            handle.write(json.dumps(row) + '\n')
        print(
            f"epoch={value['epoch']} selected={value['selected_epoch']} stale={value['controller']['stale_epochs']}",
            flush=True
        )

    start = time.perf_counter()
    result = trainer.run(event)
    trained_seconds = time.perf_counter() - start
    metrics = trainer.final_evaluate(result)
    import pickle
    with (trainer.output / 'readout.pkl').open('xb') as handle:
        pickle.dump(trainer.final_lr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    result_doc = dict(status='COMPLETE', task_id=args.task_id,
              selected_epoch=result['selected_epoch'], stop_epoch=result['checkpoint']['epoch'],
              validation_history=[row.as_dict() for row in result['history']],
              training_seconds=trained_seconds, metrics=metrics)
    result_doc['repetition'] = {k: context[k] for k in ('profile', 'run_id', 'seed', 'fold')}
    write_json(trainer.output / 'result.json', result_doc)
    print(json.dumps(metrics['final'], indent=2))


if __name__ == '__main__':
    main()
