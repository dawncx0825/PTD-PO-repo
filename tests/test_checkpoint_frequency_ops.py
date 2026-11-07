import os
"""Protect completed training telemetry across the save-frequency restart."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('frequency', Path(__file__).resolve().parents[1] / 'ops/checkpoint_frequency.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ContinuationLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        mod.R = Path(self.temp.name)
        mod.A = mod.R / 'archive'
        mod.A.mkdir()
        (mod.R / 'checkpoints').mkdir()
        self.path = mod.R / 'checkpoints/experiment_log.jsonl'
        self.prefix = [self.row(i) for i in range(1, 41)]
        self.write(mod.A / 'prefix.jsonl', self.prefix)

    @staticmethod
    def row(step):
        return {'step': step, 'actor': {'optimizer_updates': 3, 'grad_norm': .1}}

    @staticmethod
    def write(path, rows):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def test_preserves_prefix_and_resumed_updates(self):
        suffix = [{'step': 40, 'val': {'score': .2}}, self.row(41), self.row(42)]
        self.write(self.path, suffix)
        mod.merge_logs()
        _, rows = mod.read_training(self.path)
        self.assertEqual([r['step'] for r in rows], list(range(1, 43)))
        self.assertEqual(sum(r['actor']['optimizer_updates'] for r in rows), 126)
        self.assertEqual(mod.read_training(mod.A / 'prefix.jsonl')[1], self.prefix)

    def test_rejects_duplicate_and_missing_steps_without_overwriting(self):
        for bad in ([self.row(40), self.row(41)], [self.row(42)]):
            self.write(self.path, bad)
            before = self.path.read_bytes()
            with self.assertRaises(AssertionError):
                mod.merge_logs()
            self.assertEqual(self.path.read_bytes(), before)

    def test_failure_before_logger_initialization_keeps_original(self):
        self.write(self.path, self.prefix)
        before = self.path.read_bytes()
        mod.merge_logs()
        self.assertEqual(self.path.read_bytes(), before)

    def test_rejects_nonfinite_actor(self):
        row = self.row(41)
        row['actor']['grad_norm'] = float('nan')
        self.write(self.path, [row])
        with self.assertRaises(AssertionError):
            mod.merge_logs()


if __name__ == '__main__':
    unittest.main()
