import json
from pathlib import Path
import tempfile
import unittest
from scripts import checkpoint_blob as cb

class ResearchRunBackupTests(unittest.TestCase):
    def test_backup_copies_raw_metadata_and_stage_dependency_closure(self):
        from scripts.backup_research_run import backup_run
        from src.resume_checkpoint import save_checkpoint
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';(run/'raw').mkdir(parents=True)
            for name in ['config.json','prepared.json','raw/manifest.json','raw/train.jsonl','raw/validation.jsonl','raw/test.jsonl']:
                (run/name).write_text('{}\n')
            teacher=run/'teacher';contract={'identity':{'stage':'teacher'}}
            best=save_checkpoint(teacher,{'step':0,'model_state':{},'best_checkpoint':None},contract)
            save_checkpoint(teacher,{'step':3,'model_state':{},'best_checkpoint':str(Path(best).relative_to(teacher))},contract)
            factory=lambda part:cb.LocalStore(root/'remote'/part)
            status=backup_run(run,factory,root/'status.json')
            self.assertEqual(status['stages']['teacher']['remote']['step'],3)
            meta=json.loads((root/'remote/metadata/latest.json').read_text())
            snapshot=root/'remote/metadata/snapshots'/meta['snapshot']
            self.assertEqual((snapshot/'raw/train.jsonl').read_text(),'{}\n')
            restored=cb.restore_stage(factory('teacher'),root/'restored')
            self.assertTrue((restored/'checkpoints'/Path(best).name/'state.pt').exists())
            pointer_before=(root/'remote/teacher/latest.json').read_bytes()
            again=backup_run(run,factory,root/'status.json')
            self.assertEqual(again['metadata_hash'],status['metadata_hash'])
            self.assertEqual((root/'remote/teacher/latest.json').read_bytes(),pointer_before)

    def test_tuning_backup_commits_each_trial_without_copying_base_data(self):
        from scripts.backup_research_run import backup_run
        from src.resume_checkpoint import save_checkpoint
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'sweep';run.mkdir()
            (run/'plan.json').write_text(json.dumps({'trials':[{'id':'ce-1'},{'id':'kd-1'}],'provenance':{'prepared_fingerprint':'frozen-base'}}))
            for trial in ['ce-1','kd-1']:
                stage=run/'trials'/trial
                best=save_checkpoint(stage,{'step':2,'model_state':{},'best_checkpoint':None},{'identity':{'trial':trial}})
                (stage/'result.json').write_text(json.dumps({'result':{'status':'completed','final_step':2,'best_checkpoint':best}}))
            for name in ['selection.json','report.json']:(run/name).write_text('{}')
            status=backup_run(run,lambda part:cb.LocalStore(root/'remote'/part),root/'status.json')
            self.assertTrue(status['all_trials_completed'])
            self.assertEqual(set(status['trials']),{'ce-1','kd-1'})
            self.assertEqual(status['trials']['kd-1']['remote']['step'],2)
            meta=json.loads((root/'remote/metadata/latest.json').read_text())
            self.assertTrue((root/'remote/metadata/snapshots'/meta['snapshot']/'plan.json').exists())

if __name__=='__main__':unittest.main()
