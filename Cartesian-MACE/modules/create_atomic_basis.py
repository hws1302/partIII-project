# first build for the layer 1 example
# build some practice neighbourhood data
# move onto a real graph

import torch

from torch.nn import Module, Linear

from cartesian_contractions import e_rbf, cons_to_einsum, tensor_prod
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
          c_out: the Cartesian tensor rank of A
          (need to decide whether this works at c_out_max and we loop or loop outside)
          n_channels: number of different channels
          n_neighbours: number of neighbours in this given neighbourhood
          dim: dimension of features i.e. A tensor given by R^(dim * c_out)
    """

    def __init__(self, c_out: int, n_channels: int, n_neighbours: int,
                 dim: int):

        # n_neighbours is a not nice way of saying irreps in -> look to MACE for inspo
        # eventually have irreps_in as an input that can be read from a list (irrep_seq)

        super().__init__()

        # in future to map to different number of channels
        self.k = n_channels # channels out
        self.k_tilde = n_channels # channels in
        self.n_channels = n_channels
        self.c_out = c_out
        self.lin = Linear(self.k_tilde, self.k, bias=False)
        shape = [dim] * c_out
        shape.insert(0, n_channels)
        self.A = torch.zeros(shape)

        # create the tensor product, neat trick using old function
        self.einsum = cons_to_einsum(cons=[], n=self.c_out)

    def forward(self, h, rel_poss) -> torch.Tensor:

        self.dists = torch.linalg.norm(rel_poss, dim=0)
        self.norm_rel_poss = (rel_poss / self.dists)

        # mixing channels
        h = self.lin(h)

        # we are going to populate a tensor of zeros
        self.radial_emb = torch.zeros(self.n_channels, len(self.dists))
        n_values = torch.arange(1, self.n_channels + 1).unsqueeze(1)
        self.radial_emb = e_rbf(r=self.dists, n=n_values)

        rel_pos_tensors = tensor_prod(r=self.norm_rel_poss, order=c_out)

        # for i, rel_pos in enumerate(self.norm_rel_poss.T):  # fix dims
        #
        #     rel_pos_tensor = torch.einsum(
        #         self.einsum,  # the einsum required
        #         *rel_pos.repeat(self.c_out, 1)  # c_out lots of the output tensor
        #     )
        #
        #     # this is the important equation here!
        #     self.A += self.radial_emb.T[i][:, None, None] * rel_pos_tensor * h[i][:, None, None]

        self.A = torch.einsum(
            'ai,a...,ai->i...',
            self.radial_emb.T,
            rel_pos_tensors,
            h
        )


        return self.A  # shape n_channels x dim x ... c_out times ... x dim i.e. list of these As one for ea. channel


if __name__ == '__main__':

    # easier this way!
    torch.manual_seed(0)

    dim = 2  # start off in 2D to keep things simple
    n_neighbours = 5
    rel_poss = torch.randn(2, n_neighbours)
    n_channels = 100
    h = torch.randn(n_neighbours, n_channels)
    dist = torch.Tensor([1, -1])  # position (1,2)
    As = []
    c_out = 3

    A = AtomicBasis(c_out=c_out, n_channels=n_channels, n_neighbours=n_neighbours, dim=dim)
    As.append(A(h=h, rel_poss=rel_poss))

    # for c_out in range(0,3):
    #
    #     A = AtomicBasis(c_out=c_out, n_channels=n_channels, n_neighbours=n_neighbours, dim=dim)
    #     As.append(A(h=h, rel_poss=rel_poss))

    print(As)

    # running the above yields
    # as we expect we get 4 (channels) rank 2 (c_out) tensors that act as our atomic basis
    # >>> tensor([[[2.4019, -0.7598],
    #          [-0.7598, 1.5321]],
    #
    #         [[1.7606, 0.4708],
    #          [0.4708, -0.1651]],
    #
    #         [[1.2328, -7.3986],
    #          [-7.3986, -7.8531]],
    #
    #         [[3.6124, 1.7162],
    #          [1.7162, 4.1594]]], grad_fn= < AddBackward0 >)