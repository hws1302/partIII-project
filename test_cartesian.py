import unittest
import torch
import cartesian_contractions as tc

from momentmodel import CartesianContraction, CartesianEquivariantBasisBlock
from torch import tensor

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
            self.assertEqual(len(list(tc.pick_pairs(n=n, k=k))), tc.count_contractions(n=n, k=k))


    def test_equivariant_block(self):
        n, max_rank, in_dim = 4, 2, 3

        tensors_in = torch.arange(1,n*in_dim+1).reshape(n,in_dim)

        cart_equiv = CartesianEquivariantBasisBlock(n=n, max_rank=max_rank, in_dim=in_dim)

        output = cart_equiv(tensors_in=tensors_in)

        expected_output = [[tensor(8512), tensor(8350), tensor(8296)],
                             [tensor([[2240, 2464, 2688],
                                      [2560, 2816, 3072],
                                      [2880, 3168, 3456]]),
                              tensor([[2000, 2200, 2400],
                                      [2500, 2750, 3000],
                                      [3000, 3300, 3600]]),
                              tensor([[1904, 2176, 2448],
                                      [2380, 2720, 3060],
                                      [2856, 3264, 3672]]),
                              tensor([[1220, 1342, 1464],
                                      [2440, 2684, 2928],
                                      [3660, 4026, 4392]]),
                              tensor([[1169, 1336, 1503],
                                      [2338, 2672, 3006],
                                      [3507, 4008, 4509]]),
                              tensor([[1064, 1330, 1596],
                                      [2128, 2660, 3192],
                                      [3192, 3990, 4788]])]]

        assert len(expected_output) == len(output)

        for i in range(len(output)):
            for j in range(len(output[i])):
                assert torch.allclose(expected_output[i][j], output[i][j])