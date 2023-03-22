import torch
import cartesian_contractions as tc
import opt_einsum as oe

from torch import nn
from torch.nn import Parameter

"""
roughly how things work
BLOCK
^^^^
^^^^
loop over n_free indices
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


class CartesianEquivariantBasisBlock(nn.Module):
    """ "
    Block that propagates equivariants forward

    Parameters:
          nu_max: maximum tensor product order
          n_free_max: maximum number of free indices of final equivariants
          (n_free_max may become more implicit later on e.g. tensor_ranks_out)
          tensor_ranks_in: list of the ranks of the tensors in
          in_dim: in dimension of the tensors in (this will be changed at some point)

    """

    def __init__(self, nu_max: int, n_free_max: int, tensor_ranks_in: list[int], in_dim: int):
        super().__init__()

        self.n_free_max = n_free_max
        self.in_dim = in_dim  # what about examples where not all dimensions match
        self.nu_max = nu_max
        self.wsums = nn.ModuleList()
        self.tensor_ranks_in = tensor_ranks_in

        for n_free in range(0, self.n_free_max + 1):
            self.wsums.append(WeightedSum(nu_max=self.nu_max, n_free=n_free))

    def forward(self, u: torch.Tensor) -> list[torch.Tensor]:
        """ "
        Parameters:
              u: single vector input (will be changed)

        Returns:
              tensors_out: all equivariants in order of n_free
        """
        tensors_out = []

        # find invariant and equivariants upto n_free_max
        for n_free in range(0, self.n_free_max + 1):
            # does this call a different method each time (probably)
            tensors_out.append(self.wsums[n_free](u=u))

        return tensors_out

class WeightedSum(nn.Module):
    """
    Produces a weighted sum of all possible contractions with n_free indices (even if they have different numbers of contractions)
    This function calls `CartesianContraction` to carry out the heavy lifting.
    Linear combination insitialised with random weights (look at torch.nn.Linear for better intialisation route)

    Parameters:
          nu_max: tensor product order
          n_free: number of free indices of equivariants
    """
    def __init__(self, nu_max, n_free):
        super().__init__()

        self.nu_max = nu_max
        self.n_free = n_free
        self.all_tensors = [] # would be nice if we could torch-ify this bit although maybe not?
        self.contractions = nn.ModuleList()  # list of the contraction objects
        # to find this shape ahead of time
        self.n_paths = tc.count_contraction_paths(
            nu_max=self.nu_max, n_free=self.n_free
        )
        self.weights = Parameter(data=torch.randn(self.n_paths), requires_grad=True)

        # for nu in range(1, self.nu_max + 1):

        # self.contractions.append(CartesianContraction(n_free=self.n_free, nu=nu, in_dim=3))

        for nu in range(1, self.nu_max + 1):
            self.contractions.append(
                CartesianContraction(n_free=self.n_free, nu=nu, in_dim=3)
            )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
              u: single tensor in (need to change this)

        Returns:
              linear_comb: linear_combination of the tensors with n_free indices
        """
        for nu in range(1, self.nu_max + 1):  # is this iteration just wrong?
            tensors_in = u.repeat(
                nu, 1
            )  # I don't think things change if using non-identical inputs
            tensors_out = self.contractions[nu - 1](
                tensors_in=tensors_in
            )  # pick out instance

            self.all_tensors.extend(tensors_out)

            # assert len(tensors_out) == tc.count_contractions(n=nu, k=int((nu * 1 - n_free)/2))

        tensor_out_shape = [
            3 for _ in range(self.n_free)
        ]  # take out the hard coding here
        linear_comb = torch.zeros(tensor_out_shape)  # we can work out the shape before

        # explicit linear combination doesn't seem optimal
        # need to not have the tensors in a list!
        for weight, tensor in zip(self.weights, self.all_tensors):
            linear_comb += weight * tensor

        return linear_comb


class CartesianContraction(nn.Module):
    """
    Low-level class that finds the possible contractions and stores the paths for efficient use

    Parameters:
          nu: tensor product order
          n_free: number of free indices of equivariants
          in_dim: in dimension of the tensors in (this will be changed at some point)

    """

    def __init__(self, nu: int, n_free: int, in_dim: int):
        super().__init__()

        self.nu = nu
        self.n_free = n_free
        self.in_dim = in_dim
        self.buffer = {}  # pytorch object probably better
        self.einsums = []
        self.tensors_out = []
        # need to improve this to make sure ony valid combs kept
        self.k = (self.nu * 1 - self.n_free) / 2

        # so only intended calculations made
        # i.e. is n_free = 1, nu = 4, don't want any invalid combs made
        if self.k.is_integer():
            self.k = int(self.k)
            # assert self.nu * 1 - self.n_free > 0

            cons_combs = list(tc.pick_pairs(n=self.nu, k=self.k))

            # i.e. n lots of in_dim vectors
            self.shapes = nu * 1 * [(self.in_dim,)]

            for cons in cons_combs:
                einsum = tc.cons_to_einsum(cons=cons, n=nu * 1)

                self.einsums.append(
                    einsum
                )  # only doing this as einsums more instructive to read for debugging

                # self.register_buffer(einsum, oe.contract_expression(einsum, *nu * [(in_dim,)]))
                self.buffer[einsum] = oe.contract_expression(einsum, *self.shapes)

    def forward(self, tensors_in: torch.Tensor) -> list[torch.Tensor]:
        """ "
        Parameters:
              tensors_in: tensors of correct order(some degeneracy in that I have to give nu)
              (what happens if I pass in tensors of incorrect order

        Returns:
              tensors_out: all equivariants with n_free indices
        """

        for einsum in self.einsums:
            self.tensors_out.append(self.buffer[einsum](*tensors_in))

        return self.tensors_out


# class CartesianModel(nn.Module):
#
#     def __init__(self, n_layers: int, n_free_max: int, nu_max: int):
#
#         super().__init__()
#
#         self.n_layers = n_layers
#         self.n_free_max = n_free_max
#         self.nu_max = nu_max
#         self.in_dim = 3 # will sort this out at some point
#
#         vanilla_layers = nn.ModuleList()
#
#         # for i in range(self.n_layers):
#         #
#         #     if i == 0:
#         #         vanilla_layers.append(
#         #             CartesianEquivariantBasisBlock(
#         #                 nu_max=self.nu_max,
#         #                 n_free_max=self.n_free_max,
#         #                 in_dim=3
#         #             )
#         #         )
#         #
#         #     else:
#         #         vanilla_layers.append(
#         #             CartesianEquivariantBasisBlock(
#         #                 nu_max=self.nu_max,
#         #                 n_free_max=self.n_free_max,
#         #                 in_dim=3
#         #             )
#         #         )

def main():
    u = torch.randn(
        3,
    )

    n_free_max = 2
    in_dim = 3
    nu_max = 4
    n_free = 1
    tensor_ranks_in = [1]

    # wsum = WeightedSum(nu_max=nu_max, n_free=n_free)
    # print(wsum(u=u))

    block = CartesianEquivariantBasisBlock(nu_max=nu_max, n_free_max=2, tensor_ranks_in=tensor_ranks_in, in_dim=in_dim)
    print(block(u=u))
    print(block(u=u))

# use
if __name__ == "__main__":
    main()
