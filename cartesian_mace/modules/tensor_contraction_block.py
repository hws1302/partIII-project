import torch

from torch.nn import Module
from typing import List, Optional, Tuple
import opt_einsum as oe

from cartesian_mace.utils.cartesian_contractions import contraction_is_valid, pick_pairs, cons_to_einsum

class CartesianContraction(Module):
    """
    Low-level class that finds the possible contractions for a given set of free indices and stores the paths for
    efficient use

    Parameters:
          n_indices: tensor product order
          c_out: number of free indices of equivariants
          dim: in dimension of the tensors in (this will be changed at some point)

    """

    def __init__(
        self,
        n_indices: int,
        c_out: int,
        dim: int,
        n_channels: int,
        extra_dims: int,
        n_extra_dim: Optional[int] = 0,
        split: Optional[List] = None,
    ):
        super().__init__()
        self.n_indices = n_indices
        self.c_out = c_out
        self.dim = dim
        self.n_channels = n_channels
        self.buffer = {}  # can we use the torch register buffer
        self.einsums = []
        self.tensors_out = []
        self.split = split
        self.extra_dims = extra_dims
        self.n_extra_dim = (
            n_extra_dim  # extra_dim for AtomicBasis and nodes for WeightedSum
        )
        # this just applies to the A's and not the message creation step

        self.num_contractions = (self.n_indices - self.c_out) / 2

        # can only have a positive integer number of contractions
        if contraction_is_valid(num_legs_in=self.n_indices, num_legs_out=self.c_out):
            cons_combs = list(
                pick_pairs(n_indices=self.n_indices, n_contractions=self.num_contractions)
            )

            # find the tensor shapes ahead of time to store path
            # currently we do this such that a whole graph is a single contraction
            # do some testing to see the speed difference of this **
            self.shapes = self.produce_tensor_shapes(
                split=split,
                n_channels=self.n_channels,
                dim=self.dim,
                n_extra_dim=self.n_extra_dim,
            )

            for cons in cons_combs:
                einsum = cons_to_einsum(
                    cons=cons, n=n_indices, split=self.split, extra_dims=self.extra_dims
                )

                self.einsums.append(
                    einsum
                )  # only doing this as einsums more instructive to read for debugging

                # self.register_buffer(einsum, oe.contract_expression(einsum, *n_indices * [(dim,)]))
                self.buffer[einsum] = oe.contract_expression(einsum, *self.shapes)

    def produce_tensor_shapes(
        self,
        split: List[int],
        n_channels: int,
        dim: int,
        n_extra_dim: Optional[int] = 0,
    ) -> List[Tuple[int]]:
        """
        This method will find the tensor shapes (ahead of time) so that we can use `opt_einsum` to precompute the paths
        for the tensor contractions
        """

        shapes = []

        for rank in split:
            shape = ()

            if n_extra_dim:
                shape += (n_extra_dim,)

            shape += (n_channels,)

            if rank == 0:
                shape += (1,)

            else:
                shape += (dim,) * rank

            shapes.append(shape)

        return shapes

    def forward(self, tensors_in: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Returns list of all tensors of the contraction defined in __init__()


        Parameters:
              tensors_in: tensors of correct order

        Returns:
              tensors_out: all equivariants with c_out indices
        """

        tensors_out = []

        for einsum in self.einsums:
            # ** comment on shape here, things may change so look to change in future
            tensors_out.append(self.buffer[einsum](*tensors_in))

        return tensors_out