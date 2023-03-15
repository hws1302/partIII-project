import torch
import cartesian_contractions as tc
import opt_einsum as oe

from torch import nn

# for k in ks:
#
#     con(k)
#
# def con(max_rank):
#
#     for i in range(max_rank):
#
#         for k in ...

class CartesianContraction(nn.Module):

    def __init__(self, n: int, k: int, in_dim: int):
        super().__init__()

        self.n = n
        self.k = k
        # self.max_tensor_out_rank = max_tensor_out_rank
        self.in_dim = in_dim
        self.buffer = {} # pytorch object probably better


        cons_combs = list(tc.pick_pairs(n=n, k=k))

        self.einsums = []

        # i.e. n lots of in_dim vectors
        self.shapes = self.n * [(self.in_dim,)]

        for cons in cons_combs:

            einsum = tc.cons_to_einsum(cons=cons, n=n)

            self.einsums.append(einsum) # only doing this as einsums more instructive to read

        # self.register_buffer(einsum, oe.contract_expression(einsum, *n * [(in_dim,)])
            self.buffer[einsum] = oe.contract_expression(einsum, *self.shapes)

    def forward(self, tensors_in: torch.Tensor) -> list[torch.Tensor]:

        tensors_out = []

        for einsum in self.einsums:
            tensors_out.append(self.buffer[einsum](*tensors_in))

        return tensors_out

class CartesianEquivariantBasisBlock(nn.Module):

    def __init__(self, n: int, max_rank: int, in_dim: int):

        super().__init__()

        self.n = n # change the namings to be consistent
        self.max_rank = max_rank
        self.k_max = int(self.n / 2)
        self.k_min = int((self.n - self.max_rank) / 2)
        self.in_dim = in_dim

        # would be nice to have something like this
        # self.contraction = CartesianContraction(n=self.n, in_dim=)


    def forward(self, tensors_in: torch.Tensor) -> torch.Tensor:

        tensors_out = []

        for k in range(self.k_max, self.k_min - 1, -1): # python indexing need to sort

            contraction = CartesianContraction(n=self.n, k=k, in_dim=self.in_dim)

            tensors_out.append(contraction(tensors_in))

        # find all the contractions upto max_rank
        return tensors_out

if __name__ == '__main__':
        n = 6
        max_rank = 2
        in_dim = 3

        cart_equiv = CartesianEquivariantBasisBlock(n=n, max_rank=max_rank, in_dim=in_dim)

        print(cart_equiv(tensors_in=torch.arange(1,n*in_dim+1).reshape(n,in_dim)))

        # cart_con = CartesianContraction(n=4, k=2, in_dim=3)
        # print(cart_con([u,v,s,t]))

''' TO DO 
- Add in register buffer 
- Add together multiple layers 
- Add in weights like in MACE 
- Figure out the best way for ordering (i.e. where should I iterate over the possible k's


- Ask Chaitanya where he made the L-fold figures 
- Why do we need to train to get the equivariance 
- What part of the report would be instructive to start writing now? 
- Set up the test suite
- Are multiple layers here trivial i.e. tensors_in and tensors_out more complex? 
^^^ I can just look at MACE 

'''
