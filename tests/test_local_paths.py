"""Caller-controlled local storage and external replay inputs."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from agentpace import paths

class LocalPathsTest(unittest.TestCase):
    def test_local_store_and_execution_sources_are_configurable(self):
        # why: a public install must not require the developer's workspace layout.
        with tempfile.TemporaryDirectory() as root:
            base=Path(root)
            with patch.dict('os.environ',{'AGENTPACE_HOME':root,'AGENTPACE_T3_DB':str(base/'state.sqlite'),'AGENTPACE_VIGIL_LEDGER':str(base/'items.jsonl'),'AGENTPACE_CODEX_SESSIONS':str(base/'sessions'),'AGENTPACE_CLAUDE_PROJECTS':str(base/'projects')},clear=True):
                self.assertEqual(paths.home(),base)
                self.assertEqual(paths.t3_database(),base/'state.sqlite')
                self.assertEqual(paths.vigil_ledger(),base/'items.jsonl')
                self.assertEqual(paths.codex_sessions(),base/'sessions')
                self.assertEqual(paths.claude_projects(),base/'projects')
                self.assertEqual(paths.clock_timezone(),'UTC')

    def test_replay_uses_supplied_plan_and_excludes_later_source_sessions(self):
        # why: replay dates and durations are private caller inputs, never module constants.
        from experiments.deliverable_replay import replay
        from agentpace import local_runs as lr
        rows=[lr.make_row(start=1,end=2,session='early'),lr.make_row(start=3,end=4,session='cross'),lr.make_row(start=5,end=2000,session='cross')]
        result=replay([dict(asked_at=1000,actual_s=99,plan=dict(tasks=[dict(id='build',type='machine',seconds=30,basis='synthetic fixture')]))],rows)
        self.assertEqual(result[0]['train'],1)
        self.assertEqual(result[0]['p50'],30)
        self.assertEqual(result[0]['actual_s'],99)
