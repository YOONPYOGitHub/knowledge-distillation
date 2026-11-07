#!/usr/bin/env python3
"""Balanced fixed-data/fixed-Teacher validation-only smoke/pilot sweep.

No test tokenization/evaluation. The base smoke test was already viewed.
The initial --max-steps freezes the same ceiling for every CE and KD trial;
--resume must use the identical arguments. SIGTERM checkpoints return exit 75.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', required=True, type=Path,
                        help='existing frozen prepared run with completed Teacher')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='new separate sweep directory, or existing directory with --resume')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='per-trial ceiling frozen before training (default: 200)')
    parser.add_argument('--trials-file', type=Path,
                        help='strict JSON list: equal CE/KD counts, 2 to 8 trials total')
    parser.add_argument('--seed', type=int, default=42,
                        help='paired Student seed; same frozen Teacher and raw seed42 split')
    parser.add_argument('--validation-interval', type=int,
                        help='shared positive validation interval, at most max-steps')
    parser.add_argument('--patience', type=int, help='shared positive validation patience')
    parser.add_argument('--checkpoint-interval', type=int,
                        help='shared positive checkpoint interval, at most max-steps')
    args = parser.parse_args(argv)
    from src.research_tuning import _validate_trials, sweep_settings
    args.settings_overrides = {key: value for key, value in {
        'validation_interval': args.validation_interval, 'patience': args.patience,
        'checkpoint_interval_steps': args.checkpoint_interval}.items() if value is not None}
    args.trials = None
    try:
        sweep_settings({'device': 'cpu', 'batch_size': 1}, args.max_steps, 0,
                       seed=args.seed, settings_overrides=args.settings_overrides)
        if args.trials_file is not None:
            args.trials = json.loads(args.trials_file.read_text(),
                                    object_pairs_hook=_unique_fields,
                                    parse_constant=_invalid_constant)
            _validate_trials(args.trials)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return args


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON field: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('non-finite JSON constant: ' + value)


def main(argv=None, *, dependencies=None):
    args = parse_args(argv)
    from src.research_tuning import run_sweep
    return run_sweep(args.base_run, args.output_dir, resume=args.resume,
                     max_steps=args.max_steps, dependencies=dependencies,
                     trials=getattr(args, 'trials', None), seed=getattr(args, 'seed', 42),
                     settings_overrides=getattr(args, 'settings_overrides', None))


if __name__ == '__main__':
    result = main()
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] == 'completed' else 75)
