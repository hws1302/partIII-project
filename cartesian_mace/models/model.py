import torch
import math

from torch.nn import Module, Embedding, Parameter, ModuleList, ParameterList, Linear
from torch_geometric.data import Data
from typing import List, Optional
from torch_geometric.nn import global_add_pool, global_mean_pool


from cartesian_mace.modules.weighted_sum_block import WeightedSum
from cartesian_mace.modules.atomic_basis_block import AtomicBasis
from cartesian_mace.utils.cartesian_contractions import create_zero_feature_tensors


class CartesianMACE(Module):
    """
    Many-body ACE model that uses Cartesian tensors

    Creates AtomicBasis and WeightedSum modules for each layer.

    Parameters:
    n_channels: number of channels
    n_nodes: number of nodes to the central node (i.e. neighbourhood size)
    self_tp_rank_max: max rank of self tensor prod of relative positions (c1 in literature)
    basis_rank_max: max rank of the atomic basis A
    dim: dimension of positions (default = 3)
    n_layers:
    nu_max:
    feature_rank_max:

    Returns:
    h_central: updated list of feature tensors for h_central
    """

    def __init__(
        self,
        n_channels: int,
        n_nodes: int,
        self_tp_rank_max: int,
        basis_rank_max: int,
        dim: Optional[int],
        n_layers: int,
        nu_max: int,
        feature_rank_max: int,
        n_edges: int,
        pool: Optional[str] = 'sum'
    ):
        super().__init__()
        # initialise args
        self.n_channels = n_channels
        self.n_nodes = n_nodes  # could we only define this in the forward()? would help generalise network **
        self.self_tp_rank_max = self_tp_rank_max
        self.basis_rank_max = basis_rank_max
        self.dim = dim
        self.n_layers = n_layers
        self.nu_max = nu_max
        self.feature_rank_max = feature_rank_max
        self.n_edges = n_edges

        # how should I go about this, include an `atom_feature_rank` var?
        self.emb_in = Embedding(1, self.n_channels)
        self.create_atomic_bases = ModuleList()
        self.weighted_sums = ModuleList()
        self.channel_weights = ParameterList()
        self.message_weights = ParameterList()
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        for i in range(self.n_layers):

            # as in the first layer, only scalars are populated
            if i == 0:
                h_rank_max = 0
            else:
                h_rank_max = self.feature_rank_max

            # Create the atomic basis
            self.create_atomic_bases.append(
                AtomicBasis(
                    basis_rank_max=self.basis_rank_max,
                    self_tp_rank_max=self.self_tp_rank_max,
                    n_channels=self.n_channels,
                    n_nodes=self.n_nodes,
                    dim=self.dim,
                    layer=i,
                    n_edges=self.n_edges,
                    h_rank_max=h_rank_max,
                )
            )

            # Create contractions of tensor products of the atomic basis
            # maximum tensor product order `nu_max` and
            self.weighted_sums.append(
                WeightedSum(
                    nu_max=self.nu_max,
                    c_in_max=self.basis_rank_max,
                    c_out_max=self.feature_rank_max,
                    n_channels=self.n_channels,
                    dim=self.dim,
                    n_nodes=self.n_nodes,
                )
            )

            # weights for mixing the channels of the messages
            self.message_weights.append(
                Parameter(
                    data=torch.randn(
                        self.feature_rank_max + 1,
                        self.n_nodes,
                        self.n_channels,
                        self.n_channels,
                    ),
                    requires_grad=True,
                )
            )

            # weights for mixing the channels of the previous layer feature tensors
            self.channel_weights.append(
                Parameter(
                    data=torch.randn(
                        self.feature_rank_max + 1,
                        self.n_nodes, # ** remove here
                        self.n_channels,
                        self.n_channels,
                    ),
                    requires_grad=True,
                )
            )

        if False:
            # can we just flatten for when we have all irreps? **
            self.pred = torch.nn.Sequential(
                torch.nn.Linear(self.n_channels, self.n_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(self.n_channels, 2)
            )
        else:
            self.pred = torch.nn.Linear(self.n_nodes, 2)

    def update(
        self,
        channel_weights: List[torch.Tensor],
        message_weights: List[torch.Tensor],
        h: List[torch.Tensor],
        messages: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Applies MACE equation 11 to update node features.

        Separated into class method for readability of the `forward()` method

        Args:
        channel_weights (tensor): Channel weights for mixing node states from previous layers.
        message_weights (tensor): Message weights for mixing all messages of matching rank.
        h (tensor): node states.
        messages (tensor): Messages from neighbors.

        Returns:
        h_central: updated list of feature tensors for h_central
        """

        h_update = []

        # Loop through channels
        for c_out in range(self.feature_rank_max + 1):
            # Mix node states from previous layers
            h_update.append(torch.einsum(
                "ijk,ik...->ij...", channel_weights[c_out], h[c_out]
            ))  # (n_nodes x n_channels x tensor_shape) -> (n_nodes x n_channels x tensor_shape)
            # Mix all messages of matching rank
            h_update[c_out] = h_update[c_out] + torch.einsum(
                "ijk,ik...->ij...", message_weights[c_out], messages[c_out]
            )  # (n_nodes x n_channels x tensor_shape) -> (n_nodes x n_channels x tensor_shape)

        return h

    def forward(self, data: Data) -> List[torch.Tensor]:
        """
        forward method, takes in a set of positions and node feature tensors and completes a message passing step.
        currently only the central node updates its features

        data: `pytorch_geometric.data.Data` object with data for a single neighbourhood

        """
        h = create_zero_feature_tensors(
            feature_rank_max=self.feature_rank_max,
            n_nodes=self.n_nodes,
            n_channels=self.n_channels,
            dim=self.dim,
        )

        h[0] = torch.randn(self.n_nodes, self.n_channels, 1)

        # h[0] = self.emb_in(data.atoms.clone()).reshape(self.n_nodes, self.n_channels, -1)
        #
        # # sort out shaping - don't really understand how this works
        # h[0] = h0.reshape(self.n_nodes, self.n_channels, -1)

        # most important part of the class
        for create_atomic_basis, weighted_sum, channel_weights, message_weights in zip(
            self.create_atomic_bases,
            self.weighted_sums,
            self.channel_weights,
            self.message_weights,
        ):
            # h: [ n_nodes x n_channels x tensor_shape[c_1] ]

            # MACE equation 8
            # a_set: List[ n_channels x tensor_shape[c_out] ]
            a_set = create_atomic_basis(h=h, pos=data.pos, edge_index=data.edge_index)
            a_set[0] = a_set[0].reshape(
                self.n_nodes, self.n_channels, 1
            )  # currently put out incorrect shape for scalar

            # MACE equations 10 and 11
            # messages: List[ n_channels x tensor_shape[c_1] ]
            messages = weighted_sum(a_set=a_set)
            messages[0] = messages[0].reshape(
                self.n_nodes, self.n_channels, 1
            )  # same as above

            h = self.update(
                channel_weights=channel_weights,
                message_weights=message_weights,
                h=h,
                messages=messages,
            )

        # add this in **
        # i thought we only use invariants in prediction? **
        # if self.scalar_pred:
        #     # Select only scalars for prediction
        #     h = h[0]

        # either linearise now or may be easeier to use torch_scatter directly **
        # out = self.pool(h[0], data.atoms).reshape(self.n_nodes, -1)

        out = torch.sum(h[0], dim=1).T #only works for batchsize =1
        # need to add in batch.batch here **

        return self.pred(out)


if __name__ == "__main__":
    atoms = torch.zeros(4, 1).long()
    edge_index = torch.LongTensor([[0, 0, 1, 2, 1, 2, 2, 3], [1, 2, 2, 3, 0, 0, 1, 2]])
    pos = 5 * torch.Tensor(
        [[0, 0], [0.5, math.sqrt(3) * 0.5], [1, 0], [2, 0]]
    ) - torch.Tensor([5, 0])
    y = torch.Tensor([0])  # label

    data = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)

    n_channels = 3
    n_nodes = 4  # undirectional
    n_edges = len(edge_index[0])
    self_tp_rank_max = 3
    basis_rank_max = 2
    dim = 2
    n_layers = 2
    nu_max = 4
    feature_rank_max = 2

    cartesian_mace = CartesianMACE(
        n_channels=n_channels,
        n_nodes=n_nodes,
        self_tp_rank_max=self_tp_rank_max,
        basis_rank_max=basis_rank_max,
        dim=dim,
        n_layers=n_layers,
        nu_max=nu_max,
        feature_rank_max=feature_rank_max,
        n_edges=n_edges,
    )

    # do a forward pass
    print(cartesian_mace(data=data))