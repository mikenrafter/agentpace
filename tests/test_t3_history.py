"""Exact T3 labels: SQL is a read-only boundary, not an inference shortcut."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from agentpace import t3_history as history


class HistoryTest(unittest.TestCase):
    def test_exact_ids_override_ambiguous_briefs_and_split_turns(self):
        # why: two identical briefs are distinguishable by their returned run ids.
        # catches: fingerprint/time-window fallback despite an exact result id.
        runs=[{'run_id':'r1','thread_id':'t','start':1,'end':21,'actual_s':20,'status':'completed'}, {'run_id':'r2','thread_id':'t','start':30,'end':80,'actual_s':50,'status':'completed'}]
        pair={'start':2,'end':200,'expect_s':100,'features':{},'model':'sol'}
        result=history.relabel(pair,{'runId':'r1','childThreadId':'t'},runs)
        self.assertEqual(result['actual_s'],20)
        self.assertEqual(result['label_source'],'t3-run-id')
        self.assertEqual(result['run_id'],'r1')
        self.assertFalse(result['expect_is_prerun'])

    def test_concurrency_counts_only_runs_started_and_not_yet_ended(self):
        # why: future starts and completed work cannot inflate dispatch contention.
        # catches: counting the whole day's completed runs as concurrency.
        runs=[{'run_id':'a','start':1,'end':10},{'run_id':'b','start':5,'end':20},{'run_id':'c','start':10,'end':30}]
        self.assertEqual(history.concurrency(runs),{'a':0,'b':1,'c':1})

    def test_reader_does_not_create_missing_database(self):
        # why: SQLite's default connect silently creates a file; T3 must stay read-only.
        # catches: connecting without mode=ro and uri=True.
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'missing.sqlite'
            with self.assertRaises(sqlite3.OperationalError): history.read_database(path)
            self.assertFalse(path.exists())

    def test_delegate_result_child_run_id_is_an_exact_identity(self):
        # why: live delegate_task results use childRunId, while thread_send uses runId.
        # catches: indexing continuations but silently dropping initial delegates.
        payload='{"taskId":"task:a","childThreadId":"thread:a","childRunId":"run:a","status":"running"}'
        self.assertEqual(history.result_ids(payload),[{'runId':'run:a','childThreadId':'thread:a','taskId':'task:a'}])

    def test_hold_binds_to_preceding_continuation_call(self):
        # why: the old vigil export remembers delegate_task but not thread_send.
        # catches: assigning every continuation hold to the initial run id.
        rows=[{'timestamp':1,'message':{'content':[{'type':'tool_use','id':'a','name':'delegate_task','input':{'task':'initial'}}]}}, {'timestamp':3,'message':{'content':[{'type':'tool_use','id':'b','name':'t3_thread_send','input':{'threadId':'t','message':'continue'}}]}}, {'timestamp':5,'message':{'content':[{'type':'tool_use','id':'h','name':'Bash','input':{'command':'vigil hold add H --label fix --expect-s 100 --for none'}}]}}]
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'session.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            links=history.hold_index([path])
        self.assertEqual(links[(str(path),5.0,'fix')][1],3.0)

    def test_hold_in_a_command_substitution_is_indexed(self):
        # why: the orchestrator often writes `H=$(vigil hold --for none ... | python3 ...)`; the
        #      token `H=$(vigil` was not recognised, so those holds never linked to their run.
        # catches: matching only a bare `vigil` token.
        rows=[{'timestamp':1,'message':{'content':[{'type':'tool_use','id':'b','name':'t3_thread_send','input':{'threadId':'t','message':'go'}}]}}, {'timestamp':3,'message':{'content':[{'type':'tool_use','id':'h','name':'Bash','input':{'command':'H=$(vigil hold --for none --expect-s 3600 --label sol-x | python3 -c "print(1)")'}}]}}]
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'session.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            links=history.hold_index([path])
        self.assertEqual(links[(str(path),3.0,'sol-x')][2],'b')

    def test_agent_dispatch_is_indexed_only_when_asked(self):
        # why: learn links holds to background Agent subagents; the frozen round-3 replay must
        #      keep resetting at an Agent call.
        # catches: changing the default behaviour that replay relies on.
        rows=[{'timestamp':1,'message':{'content':[{'type':'tool_use','id':'a','name':'Agent','input':{'prompt':'x'}}]}}, {'timestamp':2,'toolUseResult':{'agentId':'abc'},'message':{'content':[{'type':'tool_result','tool_use_id':'a','content':'agentId: abc'}]}}, {'timestamp':3,'message':{'content':[{'type':'tool_use','id':'h','name':'mcp__vigil__hold_add','input':{'label':'x','expect_s':60}}]}}]
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'session.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(history.hold_index([path]),{})
            links=history.hold_index([path],agents=True)
            calls=history.call_index([path])
        self.assertEqual(calls[links[(str(path),3.0,'x')]],[{'agentId':'abc'}])

    def test_pair_identity_includes_source_transcript(self):
        # why: copied Claude histories contain identical hold timestamps/estimates.
        # catches: joining an original row to a different transcript's copied hold.
        raw={'file':'a','start':1,'end':2,'expect_s':10}
        stored={'session':'ca978112ca1bbdcafac2','start':1,'end':2,'expect_s':10}
        self.assertEqual(history.cohort_key(raw),history.cohort_key(stored))
        self.assertNotEqual(history.cohort_key(raw),history.cohort_key(dict(raw,file='b')))

    def test_identical_copied_holds_are_not_ambiguous_tasks(self):
        # why: one exact T3 run can appear in copied source transcripts.
        # catches: losing exact attribution for copies of the same logical hold.
        rows=[{'run_id':'r','label_source':'t3-run-id','hold_identity':'same'},{'run_id':'r','label_source':'t3-run-id','hold_identity':'same'}]
        previous=[{'label_source':'hold-fallback'}]*2
        self.assertEqual([r['label_source'] for r in history.resolve_links(rows,previous)],['t3-run-id']*2)
        rows[1]=dict(rows[1],hold_identity='different')
        self.assertEqual([r['label_source'] for r in history.resolve_links(rows,previous)],['hold-fallback']*2)

    def test_literal_database_join_extracts_derived_rows_without_writes(self):
        # why: run userMessageId must select its actual brief, not thread's latest text.
        # catches: joining on thread id, leaking brief prose, or changing the database.
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'state.sqlite';connection=sqlite3.connect(path)
            connection.executescript('CREATE TABLE orchestration_v2_projection_runs(run_id,thread_id,ordinal,provider,status,requested_at,completed_at,payload_json); CREATE TABLE orchestration_v2_projection_messages(message_id,payload_json); CREATE TABLE orchestration_v2_projection_threads(thread_id,project_id); CREATE TABLE projection_projects(project_id,workspace_root);')
            payload={'userMessageId':'m','modelSelection':{'model':'gpt-6.1-sol'},'startedAt':11,'completedAt':31}
            connection.execute('INSERT INTO orchestration_v2_projection_runs VALUES(?,?,?,?,?,?,?,?)',('r','thread:delegated-task:a',1,'codex','completed',10,31,json.dumps(payload)))
            connection.execute('INSERT INTO orchestration_v2_projection_messages VALUES(?,?)',('m',json.dumps({'text':'TOP_SECRET_BRIEF\n- Update a.py'})))
            connection.execute('INSERT INTO orchestration_v2_projection_threads VALUES(?,?)',('thread:delegated-task:a','p'))
            connection.execute('INSERT INTO projection_projects VALUES(?,?)',('p','/demo'))
            connection.commit();connection.close();before=path.read_bytes()
            rows=history.read_database(path)
            self.assertEqual(path.read_bytes(),before)
        self.assertEqual((rows[0]['actual_s'],rows[0]['model'],rows[0]['features']['items']),(20,'sol',1))
        self.assertNotIn('TOP_SECRET_BRIEF',json.dumps(rows))

    def test_reading_since_a_time_keeps_concurrency_exact(self):
        # why: learn reads T3 incrementally; the active_runs feature of a returned run must
        #      still count runs that started before the window and overlap it.
        # catches: filtering rows before computing concurrency, or no filter at all.
        from tests import learn_fixtures as fx
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'state.sqlite';fx.make_t3_db(path)
            fx.add_t3_run(path,'long',start=fx.T0+50,end=fx.T0+10_000)
            fx.add_t3_run(path,'early',start=fx.T0,end=fx.T0+40)
            fx.add_t3_run(path,'late',start=fx.T0+5_000,end=fx.T0+6_000)
            fx.add_t3_run(path,'open',start=fx.T0+5_500,end=None,status='running')
            full={r['run_id']:r for r in history.read_database(path)}
            part={r['run_id']:r for r in history.read_database(path,since=fx.T0+5_900)}
        self.assertNotIn('early',part)
        self.assertEqual({k:part[k]['features']['active_runs'] for k in ('late','long')},{k:full[k]['features']['active_runs'] for k in ('late','long')})
        self.assertEqual(part['late']['features']['active_runs'],1)

    def test_repeated_identical_tool_results_keep_one_identity(self):
        # why: Claude exports can repeat the same tool result in copied history.
        # catches: calling an exact run id ambiguous merely because it appears twice.
        call={'timestamp':1,'message':{'content':[{'type':'tool_use','id':'a','name':'delegate_task','input':{'task':'do it'}}]}}
        result={'timestamp':2,'message':{'content':[{'type':'tool_result','tool_use_id':'a','content':'{"childRunId":"r","childThreadId":"t"}'}]}}
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'session.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in (call,result,result)))
            found=history.call_index([path])
        self.assertEqual(len(found[(str(path),1.0,'a')]),1)

    def test_live_count_can_reconstruct_a_past_dispatch(self):
        # why: replaying an ETA must not substitute today's contention for past state.
        # catches: counting only rows whose final status is currently running.
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'state.sqlite';connection=sqlite3.connect(path)
            connection.execute('CREATE TABLE orchestration_v2_projection_runs(payload_json)')
            for start,end in [('1970-01-01T00:00:01Z','1970-01-01T00:00:20Z'),('1970-01-01T00:00:10Z',None)]:
                connection.execute('INSERT INTO orchestration_v2_projection_runs VALUES(?)',(json.dumps({'startedAt':start,'completedAt':end}),))
            connection.commit();connection.close()
            self.assertEqual(history.active_at(5,path),1)
            self.assertEqual(history.active_at(20,path),1)
