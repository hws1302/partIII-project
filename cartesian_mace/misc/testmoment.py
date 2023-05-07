import torch
import opt_einsum as oe

from torch import nn
from torch.nn import Parameter, ModuleDict, ParameterDict
from Cartesian_MACE.utils.cartesian_contractions import (
    find_combinations,
    pick_pairs,
    cons_to_einsum,
    count_contractions,
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
    Block that propagates equivariants forward

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

    def forward(self, u: torch.Tensor) -> list[torch.Tensor]:
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

    def __init__(self, nu_max: int, c_out: int, n_channels: int, dim: int):
        super().__init__()
        self.nu_max = nu_max
        self.c_out = c_out
        self.n_channels = n_channels
        self.tensor_bucket = []
        self.contractions = ModuleDict()  # list of the contraction objects
        self.channel_mixing = ParameterDict()  # list of weights for channel mixing
        self.tot = 0
        self.tot_list = []
        self.dim = dim

        # loop over all orders of TP
        for nu in range(1, self.nu_max + 1):
            # loop over all possible number of indices before contraction
            # e.g. [0,0,1,2,2] -> 5
            for n_indices in range(0, 2 * nu + 1):
                k = (n_indices * 1 - self.c_out) / 2

                # can only have a positive integer number of contractions
                if k.is_integer() and k > -1:
                    # find all combinations of tensors that have nu and n_indices
                    # e.g. for nu = 2 and n_indices = 2 -> [1,1] and [0,2]
                    splits = find_combinations(n=n_indices, nu=nu)

                    # record how many tensors there will be at inference (to build shapes of weights)
                    self.tot += len(splits) * count_contractions(n=n_indices, k=int(k))

                    for split in splits:
                        # gives us dict key for the contraction class
                        split_str = "".join(map(str, split))

                        # would be nice to use nn.Linear here but not sure how!**
                        self.contractions[split_str] = CartesianContraction(
                            n_indices=n_indices,
                            c_out=c_out,
                            dim=self.dim,
                            n_channels=self.n_channels,
                            split=split,
                        )
                        self.channel_mixing[split_str] = Parameter(
                            data=torch.randn(nu, self.n_channels, self.n_channels),
                            requires_grad=True,
                        )

        # instantiate the weights for mixing the paths - equivalent to summing over nu and eta_nu of B's to produce
        # messages
        self.weights = Parameter(
            data=torch.randn(self.tot, n_channels), requires_grad=True
        )

    def forward(self, a_set: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters:
              a_set: list of all the a_set over the channels (A_rank_max x n_channels)

        Returns:
              linear_comb: linear_combination of the tensors with c_out indices (n_channels x (dim)^c_out)
        """

        # very similar steps for retrieving the contraction objects and running inference
        for nu in range(1, self.nu_max + 1):
            for n_indices in range(0, 2 * nu + 1):
                splits = find_combinations(n=n_indices, nu=nu)

                k = (n_indices * 1 - self.c_out) / 2

                if k.is_integer() and k > -1:
                    for split in splits:
                        split_str = "".join(map(str, split))

                        # channel mixing of the A's
                        mix = self.channel_mixing[split_str]
                        tensors_in = [
                            torch.einsum("ij,j...->i...", mix[i], a_set[rank])
                            for i, rank in enumerate(split)
                        ]

                        tensors_out = self.contractions[split_str](
                            tensors_in=tensors_in
                        )

                        self.tensor_bucket.extend(tensors_out)

        self.tensor_bucket = torch.stack(self.tensor_bucket)

        # linear combination of all outputs of the correct tensor order
        linear_comb = torch.einsum("ij,ij...->j...", self.weights, self.tensor_bucket)

        return linear_comb, len(self.tensor_bucket), self.tot


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
        self, n_indices: int, c_out: int, dim: int, n_channels: int, split: list = None
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

        # sort this out eventually, just need the dual use for now **
        # if split:
        #     self.split = split # how are the tensors split e.g. [1,2] is a vector then matrix
        # else:
        #     self.split = [1] * n_indices # if there is not split

        self.k = (self.n_indices * 1 - self.c_out) / 2

        # can only have a positive integer number of contractions
        if self.k.is_integer() and self.k > -1:
            self.k = int(self.k)

            # assert self.n_indices * 1 - self.c_out > 0

            cons_combs = list(pick_pairs(n=self.n_indices, k=self.k))

            # gives us the correct shape of the input tensors
            # do these shape not include ellipsis dimensions

            self.shapes = [(self.n_channels,) + (self.dim,) * i for i in self.split]
            self.shapes = [
                dims
                if len(dims) != 1
                else (
                    self.n_channels,
                    1,
                )
                for dims in self.shapes
            ]

            for cons in cons_combs:
                einsum = cons_to_einsum(cons=cons, n=n_indices * 1, split=self.split)

                self.einsums.append(
                    einsum
                )  # only doing this as einsums more instructive to read for debugging

                # self.register_buffer(einsum, oe.contract_expression(einsum, *n_indices * [(dim,)]))
                self.buffer[einsum] = oe.contract_expression(einsum, *self.shapes)

    def forward(self, tensors_in: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Parameters:
              tensors_in: tensors of correct order

        Returns:
              tensors_out: all equivariants with c_out indices
        """

        for einsum in self.einsums:
            self.tensors_out.append(self.buffer[einsum](*tensors_in))

        return self.tensors_out


def main():
    # hyperparameters
    dim = 2
    nu_max = 4
    c_out_max = 2
    n_channels = 10

    # a set of many channeled atomic bases
    a_0 = torch.randn(n_channels, 1)
    a_1 = torch.randn(n_channels, dim)
    a_2 = torch.randn(n_channels, dim, dim)

    # need to generalise here to include A < 10 ish
    a_set = [a_0, a_1, a_2]

    for c_out in range(0, c_out_max + 1):
        weighted_sum = WeightedSum(
            nu_max=nu_max, c_out=c_out, n_channels=n_channels, dim=2
        )
        print(weighted_sum(a_set))


# use
if __name__ == "__main__":
    main()
