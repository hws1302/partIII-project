import torch

from torch.nn import Module, Embedding, Parameter, ModuleList, ParameterList
from torch_geometric.data import Data
from typing import List, Optional


from Cartesian_MACE.modules.momentmodel import WeightedSum
from Cartesian_MACE.modules.create_atomic_basis import AtomicBasis
from Cartesian_MACE.utils.cartesian_contractions import create_zero_feature_tensors


class CartesianMACE(Module):
    """
    Many-body ACE model that uses Cartesian tensors

    Creates AtomicBasis and WeightedSum modules for each layer.

    Parameters:
    n_channels: number of channels
    n_neighbours: number of neighbours to the central node (i.e. neighbourhood size)
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
        n_neighbours: int,
        self_tp_rank_max: int,
        basis_rank_max: int,
        dim: Optional[int],
        n_layers: int,
        nu_max: int,
        feature_rank_max: int,
    ):

        super().__init__()
        # initialise args
        self.n_channels = n_channels
        self.n_neighbours = n_neighbours
        self.self_tp_rank_max = self_tp_rank_max
        self.basis_rank_max = basis_rank_max
        self.dim = dim
        self.n_layers = n_layers
        self.nu_max = nu_max
        self.feature_rank_max = feature_rank_max

        # how should I go about this, include an `atom_feature_rank` var?
        self.emb_in = Embedding(1, self.n_channels)
        self.create_atomic_bases = ModuleList()
        self.weighted_sums = ModuleList()
        self.channel_weights = ParameterList()
        self.message_weights = ParameterList()

        # define  setting of Atomic followed by weighted contractions
        for i in range(self.n_layers):

            # Create the atomic basis
            self.create_atomic_bases.append(
                AtomicBasis(
                    basis_rank_max=self.basis_rank_max,
                    self_tp_rank_max=self.self_tp_rank_max,
                    n_channels=self.n_channels,
                    n_neighbours=self.n_neighbours,
                    dim=self.dim,
                    layer=i,
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
                )
            )

            # Create messages that can be used for updating
            self.message_weights.append(
                Parameter(
                    data=torch.randn(
                        self.feature_rank_max + 1, self.n_channels, self.n_channels
                    ),
                    requires_grad=True,
                )
            )

            self.channel_weights.append(
                Parameter(
                    data=torch.randn(
                        self.feature_rank_max + 1, self.n_channels, self.n_channels
                    ),
                    requires_grad=True,
                )
            )

    #
    def update(
        self,
        channel_weights: List[torch.Tensor],
        message_weights: List[torch.Tensor],
        h_central: List[torch.Tensor],
        messages: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Applies MACE equation 11 to update node features.

        Separated into class method for readability of the `forward()` method

        Args:
        channel_weights (tensor): Channel weights for mixing node states from previous layers.
        message_weights (tensor): Message weights for mixing all messages of matching rank.
        h_central (tensor): Central node states.
        messages (tensor): Messages from neighbors.

        Returns:
        h_central: updated list of feature tensors for h_central
        """

        # Loop through channels
        for c_out in range(self.feature_rank_max + 1):
            # Mix node states from previous layers
            h_central[c_out] = torch.einsum(
                "ij,j...->i...", channel_weights[c_out], h_central[c_out]
            )
            # Mix all messages of matching rank
            h_central[c_out] += torch.einsum(
                "ij,j...->i...", message_weights[c_out], messages[c_out]
            )

        return h_central

    def forward(
        self,
        data: Data
    ) -> List[torch.Tensor]:
        """
        forward method, takes in a set of positions and node feature tensors and completes a message passing step.
        currently only the central node updates its features

        data: `pytorch_geometric.data.Data` object with data for a single neighbourhood

        """

        # find the relative positions of atoms wrt to central node
        rel_poss = data.pos[data.edge_index[0] - data.edge_index[1]]

        # initialises zero tensors of correct shape for each of the nodes
        # shape: List[ n_neighbours x n_channels x dim**rank ]
        # rank going from 0 to feature_rank_max
        h, h_central = create_zero_feature_tensors(
            feature_rank_max=self.feature_rank_max,
            n_neighbours=self.n_neighbours,
            n_channels=self.n_channels,
            dim=self.dim,
        )

        # should be how I'm projecting along the channels later on
        # h_test = self.emb_in(data.atoms)

        # projecting atom label along channel axis (Embedding should be doing this)
        # need to figure some stuff out and change the way I have an h_central and h
        h[0] += torch.randn(self.n_neighbours, self.n_channels, 1)
        h_central[0] += torch.randn(self.n_channels, 1)

        # most important part of the class
        for create_atomic_basis, weighted_sum, channel_weights, message_weights in zip(
            self.create_atomic_bases,
            self.weighted_sums,
            self.channel_weights,
            self.message_weights,
        ):

            # h: [ n_neighbours x n_channels x tensor_shape[c_1] ]

            # MACE equation 8
            # a_set: List[ n_channels x tensor_shape[c_out] ]
            a_set = create_atomic_basis(h=h, rel_poss=rel_poss)
            a_set[0] = a_set[0].reshape(
                self.n_channels, 1
            )  # currently put out incorrect shape for scalar

            # MACE equations 10 and 11
            # messages: List[ n_channels x tensor_shape[c_1] ]
            messages = weighted_sum(a_set=a_set)
            messages[0] = messages[0].reshape(self.n_channels, 1)  # same as above

            # MACE equation 12
            h_central = self.update(
                channel_weights=channel_weights,
                message_weights=message_weights,
                h_central=h_central,
                messages=messages,
            )

        return h_central[0] # only care about the invariants in the final layer
