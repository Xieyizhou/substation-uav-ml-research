import signal
import tempfile
import unittest
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch, Mock
from scripts.vision import brightness_lr_retention as m


class IsolatedWorkers(unittest.TestCase):
    def test_cancel_cleans_only_launched_group_and_records_failure(self):
        key=m.KEYS[0]
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as s:
            out=Path(tmp);proc=Mock(pid=234567)
            proc.wait.side_effect=[KeyboardInterrupt(),0];proc.poll.side_effect=[None,0]
            s.enter_context(patch.object(m,'OUT',out));s.enter_context(patch.object(m,'ready',return_value={}))
            s.enter_context(patch.object(m,'KEYS',(key,)));s.enter_context(patch.object(m,'file_sha256',return_value='hash'))
            write=s.enter_context(patch.object(m,'frozen'));kill=s.enter_context(patch.object(m.os,'killpg'))
            s.enter_context(patch.object(m.subprocess,'Popen',return_value=proc))
            with self.assertRaises(KeyboardInterrupt):m.train()
            kill.assert_called_once_with(234567,signal.SIGTERM)
            failure=[c for c in write.call_args_list if c.args[0].name=='failure.json']
            self.assertEqual(len(failure),1)
            self.assertTrue(failure[0].args[1]['process_cleanup_confirmed'])

    def test_exhausted_attempts_do_not_spawn(self):
        key=m.KEYS[0]
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as s:
            out=Path(tmp)
            for i in range(1,4):(out/'workers'/key/f'attempt-{i:03}').mkdir(parents=True)
            s.enter_context(patch.object(m,'OUT',out));s.enter_context(patch.object(m,'ready',return_value={}))
            s.enter_context(patch.object(m,'KEYS',(key,)));spawn=s.enter_context(patch.object(m.subprocess,'Popen'))
            with self.assertRaisesRegex(ValueError,'budget exhausted'):m.train()
            spawn.assert_not_called()

    def test_unbound_completed_training_is_not_silently_reused(self):
        key=m.KEYS[0]
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as s:
            out=Path(tmp);cp=out/'training'/key/'completion.json';cp.parent.mkdir(parents=True);cp.touch()
            s.enter_context(patch.object(m,'OUT',out));s.enter_context(patch.object(m,'ready',return_value={}))
            s.enter_context(patch.object(m,'KEYS',(key,)));spawn=s.enter_context(patch.object(m.subprocess,'Popen'))
            with self.assertRaisesRegex(ValueError,'Unbound completed'):m.train()
            spawn.assert_not_called()


if __name__=='__main__':unittest.main()
