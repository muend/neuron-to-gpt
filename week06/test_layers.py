"""Focused checks for shape propagation and the 3D BatchNorm correction."""

import unittest

import torch

from experiment import make_model, shape_trace
from layers import BatchNorm1d


class LayerTests(unittest.TestCase):
    def test_tree_reduces_eight_positions(self):
        model = make_model(27, context=8, embedding=10, hidden=32,
                           hierarchical=True)
        shapes = [shape for _, shape in shape_trace(model, 8, batch_size=4)]
        self.assertIn((4, 4, 20), shapes)
        self.assertIn((4, 2, 64), shapes)
        self.assertIn((4, 32), shapes)
        self.assertEqual(shapes[-1], (4, 27))

    def test_batchnorm_reduces_batch_and_time(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        correct = BatchNorm1d(4, momentum=1.0)
        incorrect = BatchNorm1d(4, momentum=1.0, incorrect_3d=True)
        correct(x)
        incorrect(x)
        self.assertEqual(tuple(correct.last_mean.shape), (1, 1, 4))
        self.assertEqual(tuple(incorrect.last_mean.shape), (1, 3, 4))
        self.assertTrue(torch.allclose(correct.running_mean, x.mean((0, 1))))
        self.assertEqual(tuple(incorrect.running_mean.shape), (3, 4))
        correct.training = False
        self.assertEqual(tuple(correct(x[:1]).shape), (1, 3, 4))


if __name__ == "__main__":
    unittest.main()
