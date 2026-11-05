#!/usr/bin/env python3
"""Shell-free service entrypoint; preserve literal argv and auto-resume a plan."""
import os
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def command(base_run,output,max_steps,*,python=None,force_resume=False,seed=42,
            trials_file=None,validation_interval=None,patience=None,checkpoint_interval=None):
    result=[python or sys.executable,str(Path(__file__).with_name('tune_research.py')),
            '--base-run',str(base_run),'--output-dir',str(output),'--max-steps',str(max_steps)]
    for flag,value in (('--seed',seed if seed != 42 else None),
                       ('--trials-file',trials_file),
                       ('--validation-interval',validation_interval),
                       ('--patience',patience),('--checkpoint-interval',checkpoint_interval)):
        if value is not None:result.extend([flag,str(value)])
    if force_resume or (Path(output)/'plan.json').is_file():result.append('--resume')
    return result


def main():
    from scripts.tune_research import parse_args
    args=parse_args()
    from src.stage_records import reconcile_completed_stage
    reconcile_completed_stage(args.base_run,'teacher')
    argv=command(args.base_run,args.output_dir,args.max_steps,force_resume=args.resume,
                 seed=getattr(args,'seed',42),trials_file=getattr(args,'trials_file',None),
                 validation_interval=getattr(args,'validation_interval',None),
                 patience=getattr(args,'patience',None),
                 checkpoint_interval=getattr(args,'checkpoint_interval',None))
    os.execv(argv[0],argv)

if __name__=='__main__':main()
