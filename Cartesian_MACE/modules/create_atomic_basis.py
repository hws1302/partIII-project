# first build for the layer 1 example
# build some practice neighbourhood data
# move onto a real graph

import torch

from torch.nn import Module, Linear, Parameter, ModuleDict, Embedding

from Cartesian_MACE.utils.cartesian_contractions import e_rbf, cons_to_einsum, tensor_prod, contraction_is_valid

from Cartesian_MACE.modules.momentmodel import CartesianContraction
'''
GENERAL SCHEMATIC OF FIRST LAYER: 
h_0 are the scalars (usually atomic numbers) 
n.b. we only have scalars features (and vector positions) 
going into the first layer 

h_0                  r_i
│                    │
▼                    ▼
┌────────────────────┐
│       CREATE       │<-- mix channels of h
│         As         │
├─────────┬──────────┤
│         │          │
▼         ▼          ▼
0         1          2 (tensor rank)
│         │          │
▼         ▼          ▼
┌────────────────────┐
│        FIND        │ <-- mix channels of A
│    CONTRACTIONS    │ <-- TPs upto nu max
├─────────┬──────────┤
│         │          │
▼         ▼          ▼
0         1          2  (tensor rank)
│         │          │
▼         ▼          ▼
m_0       m_1        m_2 (produce message by summing over paths)
│         │          │
▼         ▼          ▼
h_0'       h_1'      h_2' (update scalar, vector and matrix features)
'''


class AtomicBasis(Module):
    """ "
    Block that creates the atomic basis vectors A for a given neighbourhood

    Parameters:
          h: feature matrix for a given neighbourhood
          relposs: the relative positions of neighbours wrt central node
          basis_rank: the Cartesian tensor rank of A
          (need to decide whether this works at basis_rank_max and we loop or loop outside)
          n_channels: number of different channels
          n_neighbours: number of neighbours in this given neighbourhood
          dim: dimension of features i.e. A tensor given by R^(dim * basis_rank)
    """

    def __init__(self, basis_rank_max: int, self_tp_rank_max: int, n_channels: int, n_neighbours: int,
                 dim: int, layer: int):

        # n_neighbours is a not nice way of saying irreps in -> look to MACE for inspo
        # eventually have irreps_in as an input that can be read from a list (irrep_seq)

        super().__init__()

        # in future to map to different number of channels
        self.n_neighbours = n_neighbours
        self.k = n_channels # channels out
        self.k_tilde = n_channels # channels in
        self.n_channels = n_channels
        self.basis_rank_max = basis_rank_max
        self.dim = dim
        # shape = [self.dim] * basis_rank # need to figure this one out **
        # shape.insert(0, n_channels)
        # self.A = torch.zeros(shape)
        self.self_tp_rank_max = self_tp_rank_max
        self.channel_weights = Parameter(data=torch.randn(self.basis_rank_max + 1, self.n_neighbours, self.n_channels, self.n_channels))
        self.contractions = ModuleDict()
        self.tensors_out = [[] for _ in range(self.basis_rank_max + 1)]
        self.extra_dims = 2 # one for the channels and the other for neighbours
        self.layer = layer

        # remove hard-coding ASAP
        if self.layer == 0:
            self.h_max_rank = 0 # have this as an input
        else:
            self.h_max_rank = 2

        for n_indices in range(0, self.h_max_rank + self.self_tp_rank_max + 1):

            # could probably add to other function but would get rid of the simplicity
            splits = find_combinations(a_max=self.h_max_rank, b_max=self.self_tp_rank_max, desired_sum=n_indices)

            for split in splits:

                split_str = "".join(map(str, split))

                for basis_rank in range(0, self.basis_rank_max + 1):

                    if contraction_is_valid(num_legs_in=n_indices, num_legs_out=basis_rank):

                        self.contractions[f'{split_str}:{basis_rank}'] = CartesianContraction(
                            n_indices=n_indices,
                            c_out=basis_rank,
                            dim=self.dim,
                            n_channels=self.n_channels,
                            extra_dims=self.extra_dims,
                            n_neighbours=n_neighbours,
                            split=split,
                        )



    def forward(self, h, rel_poss) -> torch.Tensor:

        self.dists = torch.linalg.norm(rel_poss, dim=-1)
        self.norm_rel_poss = (rel_poss / self.dists.reshape(-1,1))

        # mix channels
        # weight: n_neighbours x n_channels x n_channels
        # h[basis_rank]: n_neighbours x n_channels x tensor_shape
        # out: n_neighbours x n_channels x tensor_shape
        # can we use einops Mix() here? esp. for readability
        self.h = [
            torch.einsum('ijk,ik...->ij...', self.channel_weights[basis_rank], h[basis_rank])
            for basis_rank in range(0, self.basis_rank_max + 1)
                  ]

        # precompute to r self tensor products of normalised directions
        # do we need some kind of normalisation \propto c1?
        # each element of list has shape n_neighbours x tensor_shape
        self.r_tensors = [tensor_prod(r=self.norm_rel_poss, order=rank) for rank in range(0, self.self_tp_rank_max+1)]

        n_values = torch.arange(1, self.n_channels + 1).unsqueeze(1)
        self.radial_emb = e_rbf(r=self.dists, n=n_values).reshape(self.n_neighbours, self.n_channels)

        # ij, j...->ij...
        # pretty sure this is a dodgy expression
        r_tensors = [torch.einsum('ij,i...->ij...',self.radial_emb, self.r_tensors[basis_rank]) for basis_rank in range(0,self.self_tp_rank_max+1)]

        # now broadcast along the radial embedding

        # loop over the number of indices pre contraction
        for n_indices in range(0, self.h_max_rank + self.self_tp_rank_max + 1):

            # could probably add to other function but would get rid of the simplicity
            # maybe change the ordering here
            splits = find_combinations(a_max=self.h_max_rank, b_max=self.self_tp_rank_max, desired_sum=n_indices)

            for split in splits:

                split_str = "".join(map(str, split))

                for basis_rank in range(0, self.basis_rank_max + 1):

                    if contraction_is_valid(num_legs_in=n_indices, num_legs_out=basis_rank):

                        # h[basis_rank] shape n_channels x n_neighbours x tensor_shape
                        # r_tensors[basis_rank] shape n_neighbours x tensor_shape
                        # r_tensors gets n_channels part from the radial embedding



                        self.tensors_out[basis_rank].extend(
                            self.contractions[f'{split_str}:{basis_rank}'](
                                tensors_in=([h[split[0]], r_tensors[split[1]]])
                            )
                        )

        self.a_set = []

        # now sum along an axis
        # iterate over all the tensor ranks
        for tensor_rank_out in self.tensors_out:

            tensor_rank_out = torch.stack(tensor_rank_out)

            # sum over the neighbours axis
            # leaves us with n_channels x tensor shape

            self.a_set.append(
                torch.einsum('ijk...->k...', tensor_rank_out)
            )

        return self.a_set

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

if __name__ == '__main__':

    # easier this way!
    torch.manual_seed(0)

    dim = 2  # start off in 2D to keep things simple
    n_neighbours = 5
    rel_poss = torch.randn(n_neighbours, 2)
    n_channels = 3
    dist = torch.Tensor([1, -1])  # position (1,2)
    As = []
    basis_rank_max = 2
    self_tp_rank_max = 2 # max rank of self TP

    # figure out how the embedding works
    # emb_in Embedding(in_dim, emb_dim)

    h = []

    for basis_rank in range(0, basis_rank_max + 1):

        if basis_rank == 0:
            tensor_shape = (n_neighbours, n_channels,) + (1,)

        else:
            tensor_shape = (n_neighbours, n_channels,) + (dim,) * basis_rank

        h.append(
            torch.zeros(tensor_shape)
        )

    h

    # simulate late on layer
    h[0] += torch.randn(n_neighbours, n_channels, 1)
    # h[1] += torch.randn(n_neighbours, n_channels, 2)
    # h[2] += torch.randn(n_neighbours, n_channels, 2,2)


    A = AtomicBasis(basis_rank_max=basis_rank_max, self_tp_rank_max=self_tp_rank_max, n_channels=n_channels, n_neighbours=n_neighbours, dim=dim, layer=0)
    As.append(A(h=h, rel_poss=rel_poss))


    print(As)
