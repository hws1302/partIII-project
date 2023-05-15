import unittest
import torch
from utils.cartesian_contractions import *

# TO DO
# test the contraction class
# test the edge cases i.e. k = 0, n - 2*k < 0 etc.


class TestCartesian(unittest.TestCase):
    def setUp(self):
        pass

    def test_combinations(self):
        ns = torch.arange(0, 10)
        ks = torch.arange(0, 3)

        for n, k in zip(ns, ks):
            self.assertEqual(
                len(list(pick_pairs(n=n, k=k))), count_contractions(n=n, k=k)
            )

    # checked that weighted sum works for a simple example
    def test_weighted_sum(self):
        # first with identical and then test with non-identical version
        u = torch.arange(1, 4)
        nu_max = 4
        n_free = 1

        wsum = weighted_sum(u=u, nu_max=nu_max, n_free=1)
        # wsum.weights and then can do the inner product explicitly and still do randomly
        # also don't need to mock it this way

        # assert that it has correct number of free indices
        assert len(wsum.shape) == n_free

        print(wsum)
        assert torch.allclose(wsum, torch.tensor([1, 2, 3]))
