# first build for the layer 1 example
# build some practice neighbourhood data
# move onto a real graph

import torch

import cartesian_contractions as cc
from torch.nn import Module, Parameter

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

class create_As(Module):
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

    def __init__(self, h: torch.Tensor, rel_poss: torch.Tensor, c_out: int, n_channels: int, n_neighbours: int, dim: int):

        # n_neighbours is a not nice way of saying irreps in -> look to MACE for inspo
        # eventually have irreps_in as an input that can be read from a list (irrep_seq)
        
        super().__init__()

        # in future to map to different number of channels
        self.k = n_channels
        self.k_tilde = n_channels

        self.n_channels = n_channels
        self.c_out = c_out

        # create the random weight matrix to mix channels
        self.W = Parameter(data=torch.randn(self.k, self.k_tilde), requires_grad=True)
        self.h = h

        # create the empty matrix for our A/list of As (list of if n_channels > 1)
        self.shape = [dim] * c_out
        self.shape.insert(0, n_channels)
        self.A = torch.zeros(self.shape)


        # find the distances
        self.dists = torch.linalg.norm(rel_poss, dim=0)
        # find normalisation
        self.norm_rel_poss = (rel_poss / self.dists)  # fix transpose here

        # create the tensor product, neat trick using old function
        self.einsum = cc.cons_to_einsum(cons=[], n=self.c_out)

    def forward(self) -> torch.Tensor:

        # mixing channels
        # var name to ensure we know what is going on!
        # change back to h after implementation of torch class
        h = self.W @ self.h

        # we are going to populate a tensor of zeros
        self.radial_emb = torch.zeros(self.n_channels, len(self.dists))

        # nicer way of doing this or do I just need to stick into a fucntion
        for n in range(1, self.n_channels + 1):
            idx = n - 1
            # find nicer way to sort index vs function input
            # decide on usage of c and p parameters
            self.radial_emb[idx] = e_rbf(x=self.dists, n=n, c=2)

        for i, rel_pos in enumerate(self.norm_rel_poss.T):  # fix dims

            rel_pos_tensor = torch.einsum(
                self.einsum,  # the einsum required
                *rel_pos.repeat(self.c_out, 1)  # c_out lots of the output tensor
            )

            # this is the important equation here!
            self.A += self.radial_emb.T[i][:, None, None] * rel_pos_tensor * h.T[i][:, None, None]

        return self.A  # shape n_channels x dim x ... c_out times ... x dim i.e. list of these As one for ea. channel


def e_rbf(x: torch.Tensor, n: int, c: int) -> torch.Tensor:
    e_rbf_tilde = (2 / c) ** 0.5 * torch.sin(n * x) / x

    # add polynomial back in later
    u = 1  # - (p+1)*(p+2)/2 * x**p + p*(p+2)*x**(p+1) - p*(p+1)/2**x**(p+2)

    return e_rbf_tilde * u

if __name__ == '__main__':

    dim = 2 # start off in 2D to keep things simple
    n_neighbours = 5 # 5 neighbourhood points
    rel_poss = torch.randn(2,n_neighbours)
    n_channels = 4
    h = torch.randn(n_channels, n_neighbours)  # three channels with values (1,2,3)
    dist = torch.Tensor([1, -1])  # position (1,2)
    c_out = 2

    A = create_As(h=h, rel_poss=rel_poss, c_out=c_out, n_channels=n_channels, n_neighbours=n_neighbours, dim=dim)
    print(A())

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

    # list_As = []
    # c_out_max = 3

    # i.e. for c_out == 0 ... c_out_max
    # for c_out in range(1, c_out_max + 1):
    #     As = create_As(h=h, rel_poss=rel_poss, c_out=c_out, n_channels=n_channels, n_neighbours=n_neighbours,
    #                       dim=dim)
    #
    #     list_As.append(As())
    #
    # print(list_As)