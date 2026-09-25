import itertools
import unittest
from collections import Counter
from unittest.mock import patch
from scripts.vision.material_late_rehearsal_solver import solve
from scripts.vision import train_material_late_rehearsal as entry

class LateEntryTests(unittest.TestCase):
    def test_solver_matches_small_exhaustive_optimum(self):
        for seq,src in [(['a','b','c','a','a','a'],dict(a='x',b='y',c='z')),
                        (['a','b','c','d','a','a'],dict(a='x',b='x',c='y',d='z'))]:
            g=len(set(src.values()))
            feasible=[p for p in set(itertools.permutations(seq)) if len({src[m] for m in p[-g:]})==g]
            best=min(sum(a!=b for a,b in zip(seq,p)) for p in feasible)
            self.assertEqual(solve(seq,src,7)['minimum_changed_slots'],best)
    def test_default_never_trains(self):
        with patch.object(entry,'ready') as ready,patch.object(entry,'run') as run,patch.object(entry,'worker') as worker:
            entry.main([]);ready.assert_called_once();run.assert_not_called();worker.assert_not_called()
    def test_worker_needs_explicit_training(self):
        with self.assertRaises(SystemExit):entry.main(['--worker','L-7'])
    def test_unknown_cell(self):
        with self.assertRaises(ValueError):entry.contract('L-99')

if __name__=='__main__':unittest.main()
