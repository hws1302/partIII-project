# first build for the layer 1 example
# build some practice neighbourhood data
# move onto a real graph

import torch
import math

from torch.nn import Module, Parameter, ModuleDict
from cartesian_mace.utils.cartesian_contractions import (
    e_rbf,
    tensor_prod,
    contraction_is_valid,
    create_zero_feature_tensors,
)
from cartesian_mace.modules.tensor_contraction_block import CartesianContraction
from torch_scatter import scatter


class AtomicBasis(Module):
    """ "
    Block that creates the atomic basis vectors A for a given neighbourhood

    Parameters:
          h: feature matrix for a given neighbourhood
          relposs: the relative positions of neighbours wrt central node
          basis_rank: the Cartesian tensor rank of A
          (need to decide whether this works at basis_rank_max and we loop or loop outside)
          n_channels: number of different channels
          n_nodes: number of nodes in the graph
          dim: dimension of features i.e. A tensor given by R^(dim * basis_rank)
    """

    def __init__(
        self,
        basis_rank_max: int,
        self_tp_rank_max: int,
        n_channels: int,
        n_nodes: int,
        dim: int,
        layer: int,
        n_edges: int, # need to remove
        h_rank_max: int,
    ):
        # n_neighbours is a not nice way of saying irreps in -> look to MACE for inspo
        # eventually have irreps_in as an input that can be read from a list (irrep_seq)

        super().__init__()

        # in future to map to different number of channels
        self.n_nodes = n_nodes
        self.k = n_channels  # channels out
        self.k_tilde = n_channels  # channels in
        self.n_channels = n_channels
        self.basis_rank_max = basis_rank_max
        self.dim = dim
        self.n_edges = n_edges
        self.h_rank_max = h_rank_max
        # shape = [self.dim] * basis_rank # need to figure this one out **
        # shape.insert(0, n_channels)
        # self.A = torch.zeros(shape)
        self.self_tp_rank_max = self_tp_rank_max
        self.channel_weights = Parameter(
            data=torch.randn(
                self.h_rank_max + 1, self.n_channels, self.n_channels
            )
        )
        self.contractions = ModuleDict()
        self.tensors_out = [[] for _ in range(self.basis_rank_max + 1)]
        self.extra_dims = 2  # one for the channels and the other for neighbours
        self.layer = layer


        for n_indices in range(0, self.h_rank_max + self.self_tp_rank_max + 1):
            # could probably add to other function but would get rid of the simplicity
            splits = find_combinations(
                a_max=self.h_rank_max,
                b_max=self.self_tp_rank_max,
                desired_sum=n_indices,
            )

            for split in splits:
                split_str = "".join(map(str, split))

                for basis_rank in range(0, self.basis_rank_max + 1):
                    if contraction_is_valid(
                        num_legs_in=n_indices, num_legs_out=basis_rank
                    ):
                        self.contractions[
                            f"{split_str}:{basis_rank}"
                        ] = CartesianContraction(
                            n_indices=n_indices,
                            c_out=basis_rank,
                            dim=self.dim,
                            n_channels=self.n_channels,
                            extra_dims=self.extra_dims,
                            n_extra_dim=n_edges,  # switched out from n_neighbours **
                            split=split,
                        )

    def forward(self, h, pos, edge_index) -> torch.Tensor:
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
        self.dists = torch.linalg.norm(rel_pos, dim=-1)
        self.norm_rel_pos = rel_pos / self.dists.reshape(-1, 1)
        self.n_edges = len(edge_index[0])

        # mix channels
        # weight: n_neighbours x n_channels x n_channels
        # h[basis_rank]: n_neighbours x n_channels x tensor_shape
        # out: n_neighbours x n_channels x tensor_shape
        # can we use einops Mix() here? esp. for readability

        # expand h needs better explanation **
        h = [g[edge_index[1]] for g in h]

        self.h = [
            torch.einsum( # [n_channels, n_channels] x [n_edges, n_channels, tensor_shape] -> [n_edges x n_channels x tensor shape]
                "ij,kj...->ki...", self.channel_weights[h_rank], h[h_rank]
        )
            for h_rank in range(0, self.h_rank_max + 1)
        ]

        # precompute to r self tensor products of normalised directions
        # do we need some kind of normalisation \propto c1?
        # each element of list has shape n_neighbours x tensor_shape
        self.r_tensors = [
            tensor_prod(r=self.norm_rel_pos, order=rank)
            for rank in range(0, self.self_tp_rank_max + 1)
        ]

        n_values = torch.arange(1, self.n_channels + 1).unsqueeze(1)
        # why are all channels the same value
        # self.dists is r_ij where i is edge_index 0 and j is edge_index 1
        self.radial_emb = e_rbf(r=self.dists, n=n_values).reshape(
            self.n_edges, self.n_channels
        )

        # ij, j...->ij...
        # pretty sure this is a dodgy expression
        r_tensors = [
            torch.einsum("ij,i...->ij...", self.radial_emb, self.r_tensors[basis_rank])
            for basis_rank in range(0, self.self_tp_rank_max + 1)
        ]

        # now broadcast along the radial embedding

        # loop over the number of indices pre contraction
        for n_indices in range(0, self.h_rank_max + self.self_tp_rank_max + 1):
            # could probably add to other function but would get rid of the simplicity
            # maybe change the ordering here
            splits = find_combinations(
                a_max=self.h_rank_max,
                b_max=self.self_tp_rank_max,
                desired_sum=n_indices,
            )

            for split in splits:
                split_str = "".join(map(str, split))

                for basis_rank in range(0, self.basis_rank_max + 1):
                    if contraction_is_valid(
                        num_legs_in=n_indices, num_legs_out=basis_rank
                    ):
                        # h[basis_rank] shape n_channels x n_neighbours x tensor_shape
                        # r_tensors[basis_rank] shape n_neighbours x tensor_shape
                        # r_tensors gets n_channels part from the radial embedding

                        self.tensors_out[basis_rank].extend(
                            self.contractions[f"{split_str}:{basis_rank}"](
                                tensors_in=([h[split[0]], r_tensors[split[1]]])
                            )
                        )

        self.a_set = []
        self.a_set_pyg = []

        # now sum along an axis
        # iterate over all the tensor ranks
        for tensor_rank_out in self.tensors_out:

            if tensor_rank_out:
                tensor_rank_out = torch.stack(tensor_rank_out)

            # sum over the neighbours axis
            # leaves us with n_channels x tensor shape

                self.a_set.append(torch.einsum("ijk...->k...", tensor_rank_out))

                # # need to sum over both the neighbours (scatter)
                # # sum over different contractions (we don't weight ?! -> we should do implicitly via the radial embedding)
                tensor_rank_out = torch.einsum("i...->...", tensor_rank_out)
                #
                # # sum over the different neighbourhoods
                # # indexing of first channels n_nodes is done in accordance with edge_index[0]
                # # [n_edges x n_channels x tensor_shape] -> [n_nodes x n_channels x tensor_shape]
                self.a_set_pyg.append(
                    scatter(src=tensor_rank_out, index=edge_index[0], dim=0, reduce="sum")
                )

        return self.a_set_pyg


def find_combinations(a_max, b_max, desired_sum):
    combinations = []

    if desired_sum > a_max + b_max or desired_sum < 0:
        return combinations

    a_min, b_min = 0, 0
    a_start = max(a_min, desired_sum - b_max)
    a_end = min(a_max, desired_sum - b_min)

    for a in range(a_start, a_end + 1):
        b = desired_sum - a
        combinations.append((a, b))

    return combinations


if __name__ == "__main__":
    # easier this way!
    torch.manual_seed(0)

    dim = 2  # start off in 2D to keep things simple

    n_channels = 3
    As = []
    basis_rank_max = 2
    self_tp_rank_max = 2  # max rank of self TP

    # going both ways
    # edge_index = torch.LongTensor([[0, 0, 1, 2], [1, 2, 2, 3]])
    edge_index = torch.LongTensor([[0, 0, 1, 2, 1, 2, 2, 3], [1, 2, 2, 3, 0, 0, 1, 2]])
    pos = 5 * torch.Tensor(
        [[0, 0], [0.5, math.sqrt(3) * 0.5], [1, 0], [2, 0]]
    ) - torch.Tensor([6, 2])
    # print(pos)

    n_nodes = len(edge_index[0])

    # pos = torch.Tensor([
    #     [0,0],
    #     [1,1],
    #     [3,4],
    #     [-2,3],
    # ])
    #
    # edge_index = torch.LongTensor([[0, 0, 0], [1, 2, 3]])

    h = []

    h = create_zero_feature_tensors(
        feature_rank_max=basis_rank_max, n_nodes=n_nodes, n_channels=n_channels, dim=dim
    )

    # simulate late on layer
    h[0] += torch.randn(n_nodes, n_channels, 1)
    # h[1] += torch.randn(n_neighbours, n_channels, 2)
    # h[2] += torch.randn(n_neighbours, n_channels, 2,2)

    A = AtomicBasis(
        basis_rank_max=basis_rank_max,
        self_tp_rank_max=self_tp_rank_max,
        n_channels=n_channels,
        n_nodes=n_nodes,
        dim=dim,
        layer=0,
    )
    As.append(A(h=h, pos=pos, edge_index=edge_index))

    print(As)