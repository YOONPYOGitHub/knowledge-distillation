from pathlib import Path
from scripts import tuning_service


def test_service_fresh_has_no_empty_argument(tmp_path):
    command=tuning_service.command(tmp_path/'base',tmp_path/'new',200,python='/python')
    assert '' not in command
    assert '--resume' not in command
    assert command[-2:]==['--max-steps','200']


def test_service_rejects_inconsistent_teacher_before_exec(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import pytest
    import scripts.tune_research as cli
    import src.stage_records as records
    args=SimpleNamespace(base_run=tmp_path/'base',output_dir=tmp_path/'out',max_steps=200,resume=False)
    monkeypatch.setattr(cli,'parse_args',lambda:args)
    calls=[]
    def refuse(directory,stage):
        calls.append((directory,stage))
        raise ValueError('inconsistent Teacher record')
    monkeypatch.setattr(records,'reconcile_completed_stage',refuse)
    monkeypatch.setattr(tuning_service.os,'execv',lambda *argv:calls.append('exec'))
    with pytest.raises(ValueError,match='Teacher'):
        tuning_service.main()
    assert calls==[(args.base_run,'teacher')]


def test_service_adds_resume_only_for_existing_plan(tmp_path):
    output=tmp_path/'existing';output.mkdir();(output/'plan.json').write_text('{}')
    command=tuning_service.command(tmp_path/'base',output,200,python='/python')
    assert command.count('--resume')==1
    assert '' not in command
