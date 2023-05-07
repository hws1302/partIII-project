import torch
from typing import List, Optional
from torch.nn import Module, Parameter, ModuleDict, ParameterDict
from cartesian_mace.modules.tensor_contraction_block import CartesianContraction
from cartesian_mace.utils.cartesian_contractions import find_combinations, count_contractions, contraction_is_valid


class WeightedSum(Module):
    """
    Produces a weighted sum of all possible contractions with c_out indices
    (even if they have different numbn` ers of contractions)
    This function calls `CartesianContractioto carry out the heavy lifting.
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
        n_extra_dims: Optional[int] = 2
    ):
        super().__init__()
        self.n_extra_dims = n_extra_dims
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
                splits = find_combinations(n=n_indices, nu=nu, c_max=c_in_max)

                for split in splits:
                    # gives us dict key for the contraction class
                    # acts as unique hash
                    split_str = "".join(map(str, split))

                    # only one mix of A's channels per split (i.e. same mix for all `c_out`)
                    # currently create the mixing before checking if there are any valid one **
                    # one channel mixing matrix per term in the prod
                    # [nu, n_channels, n_channels]
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
                                n=n_indices, n_contractions=num_contractions
                            )

                            self.contractions[
                                f"{split_str}:{c_out}"
                            ] = CartesianContraction(
                                n_indices=n_indices,
                                c_out=c_out,
                                dim=self.dim,
                                n_channels=self.n_channels,
                                n_extra_dim=self.n_extra_dims,  # need to put in size for contractions ahead of time can we get past this?**
                                # alternative would be to create the contraction and call it once per node in the forward
                                # (we can disucss this)
                                split=split,
                            )

        # add in extra dim for the node dimension
        # need to normalise these for ea. channel


        self.path_weights = [
            Parameter(
                data=torch.randn(self.tot[c_out], self.n_channels), requires_grad=True
            ) # [n_paths, n_channels]
            for c_out in range(0, self.c_out_max + 1)
        ]



        # self.path_weights = []
        # for c_out in range(0, self.c_out_max + 1):
        #
        #     weights = torch.randn(self.tot[c_out], self.n_channels)
        #     weights /= weights.std(dim=0)
        #     self.path_weights.append(
        #         Parameter(
        #             data=weights, requires_grad=True
        #         )
        #     )

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
            for n_indices in range(0, self.c_in_max * nu + 1):
                splits = find_combinations(n=n_indices, nu=nu, c_max=self.c_in_max)

                for split in splits:
                    split_str = "".join(map(str, split))

                    # channel mixing of the A's
                    # for A's of ea. rank we have with tensor_shape = [2,2] for C=2
                    # in: [n_channels, n_channels] x [n_nodes, n_channels, tensor_shape]
                    # out: [n_nodes, n_channels, tensors_shape]
                    mix = self.channel_mixing[split_str]
                    tensors_in = [
                        torch.einsum("ij,kj...->ki...", mix[i], a_set[rank])
                        for i, rank in enumerate(split)
                    ]

                    # tensors_in = [a_set[rank] for rank in split]

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
        # again we deal with each tensor rank separately
        # in: [n_paths, n_channels] x [n_paths, n_nodes, n_channels, tensor_shape]
        for c_out in range(0, self.c_out_max + 1):
            messages.append(
                torch.einsum(
                    "ij,iaj...->aj...",
                    self.path_weights[c_out],
                    torch.stack(tensor_bucket[c_out]),
                )
            )

        return messages