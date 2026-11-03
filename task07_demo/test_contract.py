"""Tests for leakage protection and evaluation semantics, not training quality."""
import unittest
import numpy as np
from common import SPLIT,validate_split,binary_metrics
class ContractTests(unittest.TestCase):
    def test_patient_leakage_rejected(self):
        bad={k:list(v) for k,v in SPLIT.items()};bad['test'][0]=bad['train'][0]
        with self.assertRaises(ValueError):validate_split(bad)
    def test_dice_and_absence(self):
        g=np.array([1,1,0,0],bool);p=np.array([1,0,1,0],bool)
        self.assertEqual(binary_metrics(g,p)['dice'],.5)
        empty=np.zeros(4,bool);self.assertEqual(binary_metrics(empty,empty)['dice'],1.)
        self.assertEqual(binary_metrics(g,empty)['dice'],0.)
        self.assertIsNone(binary_metrics(g,empty)['precision'])
    def test_task_labels_and_split_counts(self):
        validate_split(SPLIT);self.assertEqual([len(SPLIT[k]) for k in ('train','val','test')],[8,2,2])
if __name__=='__main__':unittest.main()
