import torch
import opt_einsum as oe

from torch import nn
from typing import Optional, List, Tuple
from torch.nn import Parameter, ModuleDict, ParameterDict
from Cartesian_MACE.utils.cartesian_contractions import (
    find_combinations,
    pick_pairs,
    cons_to_einsum,
    count_contractions,
    contraction_is_valid,
)


"""
roughly how things work
BLOCK
^^^^
^^^^
loop over c_out indices
^^^^
^^^^
WEIGHTED SUM
^^^^
^^^^
loop over TP order
^^^^
^^^^
CONTRACTION
^^^^
^^^^
find all contractions for mapping:
inputs, TP order -> n contractions to given tensor rank
"""


############## IGNORE THIS CLASS FOR NOW ##############
class CartesianEquivariantBlock(nn.Module):
    """
    Blocnum_contractionsthat propagates equivariants forward

    Parameters:
          nu_max: maximum tensor product order
          c_out_max: maximum number of free indices of final equivariants
          (c_out_max may become more implicit later on e.g. tensor_ranks_out)
          tensor_ranks_in: list of the ranks of the tensors in
          dim: in dimension of the tensors in (this will be changed at some point)

    """

    def __init__(self, nu_max: int, c_out_max: int, dim: int):
        super().__init__()

        self.nu_max = nu_max
        self.c_out_max = c_out_max
        self.dim = dim

        self.weighted_sums = nn.ModuleList()

        for c_out in range(0, self.c_out_max + 1):
            self.weighted_sums.append(
                WeightedSum(
                    nu_max=self.nu_max,
                    c_out=c_out,
                    n_channels=self.n_channels,
                    dim=self.dim,
                )
            )

    def forward(self, u: torch.Tensor) -> List[torch.Tensor]:
        """ "
        Parameters:
              u: single vector input (will be changed)

        Returns:
              tensors_out: all equivariants in order of c_out
        """
        tensors_out = []

        # find invariant and equivariants upto c_out_max
        for c_out in range(0, self.c_out_max + 1):
            # does this call a different method each time (probably)
            tensors_out.append(self.weighted_sums[c_out](u=u))

        return tensors_out


# change class name?
class WeightedSum(nn.Module):
    """
    Produces a weighted sum of all possible contractions with c_out indices
    (even if they have different numbers of contractions)
    This function calls `CartesianContraction` to carry out the heavy lifting.
    Linear combination initialised with random weights (look at torch.nn.Linear for better intialisation route)

    Parameters:
          nu_max: tensor product order
          c_out: number of free indices of equivariants
    """

    def __init__(
        self,
        nu_max: int,
        c_in_max: int,
        c_out_max: int,
        n_channels: int,
        dim: int,
        n_nodes: int,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.nu_max = nu_max
        self.c_in_max = c_in_max  # max rank of A's
        self.c_out_max = c_out_max
        self.n_channels = n_channels
        self.contractions = ModuleDict()  # list of the contraction objects
        self.channel_mixing = ParameterDict()  # list of weights for channel mixing
        self.tot = [
            0,
        ] * (
            c_out_max + 1
        )  # running total of number of each set of contractions
        self.dim = dim

        # loop over all orders of TP
        # i.e. have nu lots of A elements
        for nu in range(1, self.nu_max + 1):
            # loop over all possible number of indices before contraction
            # this comes from the fact that A's can have different rank
            # e.g. for nu=3 n_indices=5 an example of a split -> [1,2,2]
            for n_indices in range(0, c_in_max * nu + 1):
                # find all combinations of tensors that have nu and n_indices
                # e.g. for nu = 2 and n_indices = 2 -> [1,1] and [0,2]
                splits = find_combinations(n=n_indices, nu=nu)

                for split in splits:
                    # gives us dict key for the contraction class
                    # acts as unique hash
                    split_str = "".join(map(str, split))

                    # only one mix of A's channels per split (i.e. same mix for all `c_out`)
                    # currently create the mixing before checking if there are any valid one **
                    self.channel_mixing[split_str] = Parameter(
                        data=torch.randn(nu, self.n_channels, self.n_channels),
                        requires_grad=True,
                    )

                    for c_out in range(0, self.c_out_max + 1):
                        num_contractions = (n_indices - c_out) / 2

                        # can only have a positive integer number of contractions
                        if contraction_is_valid(
                            num_legs_in=n_indices, num_legs_out=c_out
                        ):
                            # record how many tensors there will be at inference (to build shapes of weights)
                            self.tot[c_out] += count_contractions(
                                n=n_indices, num_contractions=num_contractions
                            )

                            self.contractions[
                                f"{split_str}:{c_out}"
                            ] = CartesianContraction(
                                n_indices=n_indices,
                                c_out=c_out,
                                dim=self.dim,
                                n_channels=self.n_channels,
                                extra_dims=1,  # take out hard-coding
                                n_extra_dim=self.n_nodes,  # need to put in size for contractions ahead of time can we get past this?**
                                split=split,
                            )

        # add in extra dim for the node dimension
        self.path_weights = [
            Parameter(
                data=torch.randn(self.tot[c_out], self.n_channels), requires_grad=True
            )
            for c_out in range(0, self.c_out_max + 1)
        ]

    def forward(self, a_set: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters:
              a_set: list of all the a_set over the channels (A_rank_max x n_channels)

        Returns:
              messsage: message formed from a linear combination of the tensors with c_out indices
              (n_channels x (dim)^c_out)
        """
        # one list per c_out rank
        messages = []
        tensor_bucket = [[] for _ in range(0, self.c_out_max + 1)]

        # very similar steps for retrieving the contraction objects and running inference
        for nu in range(1, self.nu_max + 1):
            for n_indices in range(0, 2 * nu + 1):
                splits = find_combinations(n=n_indices, nu=nu)

                for split in splits:
                    split_str = "".join(map(str, split))

                    # channel mixing of the A's
                    # need to add in extra dim for nodes here
                    mix = self.channel_mixing[split_str]
                    tensors_in = [
                        torch.einsum("ij,kj...->ki...", mix[i], a_set[rank])
                        for i, rank in enumerate(split)
                    ]

                    for c_out in range(0, self.c_out_max + 1):
                        num_contractions = (n_indices - c_out) / 2

                        if contraction_is_valid(
                            num_legs_in=n_indices, num_legs_out=c_out
                        ):
                            tensors_out = self.contractions[f"{split_str}:{c_out}"](
                                tensors_in=tensors_in
                            )

                            tensor_bucket[c_out].extend(tensors_out)

        # for ea. c_out rank we take the linear weighted combination of paths
        for c_out in range(0, self.c_out_max + 1):
            messages.append(
                torch.einsum(
                    "ij,iaj...->aj...",
                    self.path_weights[c_out],
                    torch.stack(tensor_bucket[c_out]),
                )
            )

        return messages


class CartesianContraction(nn.Module):
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
                pick_pairs(n=self.n_indices, num_contractions=self.num_contractions)
            )

            # find the tensor shapes ahead of time to store path
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
        Parameters:
              tensors_in: tensors of correct order

        Returns:
              tensors_out: all equivariants with c_out indices
        """

        tensors_out = []

        for einsum in self.einsums:
            # self.tensors_out.append(
            #     torch.einsum(einsum, *tensors_in)
            # )
            # stopped working after added multiple nodes aswell as multiple channels
            tensors_out.append(self.buffer[einsum](*tensors_in))

        return tensors_out


# use
if __name__ == "__main__":
    # hyperparameters
    dim = 2
    nu_max = 4
    c_in_max = 2
    c_out_max = 2
    n_channels = 3
    n_nodes = 4

    # a set of many channeled atomic bases
    a_0 = torch.randn(n_nodes, n_channels, 1)
    a_1 = torch.randn(n_nodes, n_channels, dim)
    a_2 = torch.randn(n_nodes, n_channels, dim, dim)

    # need to generalise here to include A < 10 ish
    a_set = [a_0, a_1, a_2]

    weighted_sum = WeightedSum(
        nu_max=nu_max,
        c_in_max=c_in_max,
        c_out_max=c_out_max,
        n_channels=n_channels,
        dim=dim,
        n_nodes=n_nodes,
    )
    print(weighted_sum(a_set))
