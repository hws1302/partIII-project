import torch
import math
import cartesian_contractions as tc
import opt_einsum as oe

from torch import nn
from torch.nn import Parameter

# for k in ks:
#
#     con(k)
#
# def con(max_rank):
#
#     for i in range(max_rank):
#
#         for k in ...

# class CartesianContraction(nn.Module):
#
#     def __init__(self, k: int, shapes_in: tuple, nu_max: int):
#
#         super().__init__()
#
#         self.nu_max = nu_max
#         self.shapes_in = shapes_in
#         self.k = k
#         # self.max_tensor_out_rank = max_tensor_out_rank
#         # self.in_dim = in_dim
#         self.buffers = [{} for _ in range(self.nu_max)]  # pytorch object probably better
#
#         for nu in range(1, self.nu_max + 1):
#
#             # i.e indices in * tensor prod order
#             # or len(shapes_in)
#             n = len(self.shapes_in) * nu
#
#             cons_combs = list(tc.pick_pairs(n=n, k=k))
#
#             self.einsums = []
#
#             # i.e. n lots of in_dim vectors
#             # this shapes_in[1] probably only works for vectors but can change later
#             self.shapes = n * [(self.shapes_in[1],)]
#
#             for cons in cons_combs:
#                 # how do we split up the einsum in a nice way
#                 einsum = tc.cons_to_einsum(cons=cons, n=n)
#
#                 self.einsums.append(einsum)  # only doing this as einsums more instructive to read
#
#                 # self.register_buffer(einsum, oe.contract_expression(einsum, *n * [(in_dim,)])
#                 # sort of (nu - 1) index here
#                 self.buffers[nu - 1][einsum] = oe.contract_expression(einsum, *self.shapes)
#
#     def forward(self, tensors_in: torch.Tensor) -> list[torch.Tensor]:
#
#         tensors_out = [[] for _ in range(self.nu_max)]
#
#         for nu in range(1, self.nu_max + 1):
#
#             for einsum in self.einsums:
#                 tensors_out[nu - 1].append(self.buffers[nu - 1][einsum](*tensors_in))
#
#         return tensors_out


class CartesianEquivariantBasisBlock(nn.Module):

    def __init__(self, nu_max: int, n_free_max: int, in_dim: int):
        super().__init__()

        self.n_free_max = n_free_max
        self.in_dim = in_dim # what about examples where not all dimensions match
        self.nu_max = nu_max

        # would be nice to have something like this
        # self.contraction = CartesianContraction(n=self.n, in_dim=)

    def forward(self, u: torch.Tensor) -> list[torch.Tensor]:

        tensors_out = []

        for n_free in range(0, n_free_max + 1):

            print(n_free)

            # would be nicer if we would instantiate for all k at class initialisation
            wsum = WeightedSum(nu_max=self.nu_max, n_free=n_free)

            u = torch.randn(3,)
            tensors_out.append(wsum(u=u))

            # wsum = WeightedSum(nu_max=nu_max, n_free=n_free)
            # #
            # print(wsum(u=u))

        # find all the contractions upto n_free_max
        return tensors_out

class CartesianContraction(nn.Module):

    def __init__(self, nu: int, n_free: int, in_dim: int):
        super().__init__()



        self.nu = nu
        self.n_free = n_free
        # * 1 as we only deal with a vector input for now!
        # self.k = int(nu*1 - n_free)
        # self.max_tensor_out_rank = max_tensor_out_rank
        self.in_dim = in_dim
        self.buffer = {} # pytorch object probably better
        # {nu :{} for nu in range(1,nu_max+1)}
        self.einsums = []

        # need to improve this to make sure ony valid combs kept
        self.k = (self.nu * 1 - self.n_free)/2

        # so only intended calculations made
        # i.e. is n_free = 1, nu = 4, don't want any invalid combs made
        if self.k.is_integer():

            self.k = int(self.k)
            # assert self.nu * 1 - self.n_free > 0

            cons_combs = list(tc.pick_pairs(n=self.nu, k=self.k))

            # i.e. n lots of in_dim vectors
            self.shapes = nu * 1 * [(self.in_dim,)]

            for cons in cons_combs:

                einsum = tc.cons_to_einsum(cons=cons, n=nu*1)

                self.einsums.append(einsum) # only doing this as einsums more instructive to read

                # self.register_buffer(einsum, oe.contract_expression(einsum, *nu * [(in_dim,)]))
                self.buffer[einsum] = oe.contract_expression(einsum, *self.shapes)

    def forward(self, tensors_in: torch.Tensor) -> list[torch.Tensor]:

        self.tensors_out = []

        for einsum in self.einsums:

            self.tensors_out.append(self.buffer[einsum](*tensors_in))

        return self.tensors_out


class WeightedSum(nn.Module):

    def __init__(self, nu_max, n_free):

        super().__init__()

        self.nu_max = nu_max
        self.n_free = n_free

        self.all_tensors = []  # would be nice if we could torch-ify this bit

    def forward(self, u: torch.Tensor) -> torch.Tensor:

        for nu in range(1, self.nu_max + 1):  # is this iteration just wrong?
            tensors_in = u.repeat(nu, 1)  # I don't think things change if using non-identical inputs
            contraction = CartesianContraction(n_free=self.n_free, nu=nu, in_dim=3)  # need to change in_dim at some point
            # now we need to get the weighted sum

            tensors_out = contraction(tensors_in=tensors_in)

            self.all_tensors.extend(tensors_out)

            # assert len(tensors_out) == tc.count_contractions(n=nu, k=int((nu * 1 - n_free)/2))


        tensor_out_shape = [3 for _ in range(self.n_free)] # take out the hard coding here
        linear_comb = torch.zeros(tensor_out_shape)  # we can work out the shape before
        self.weights = Parameter(data=torch.randn(len(self.all_tensors)), requires_grad=True)

        # explicit linear combination doesn't seem optimal
        # need to not have the tensors in a list!
        for weight, tensor in zip(self.weights, self.all_tensors):
            linear_comb += weight * tensor

        return linear_comb

if __name__ == '__main__':

    u = torch.randn(3,)

    n_free_max = 2
    in_dim = 3
    nu_max = 4
    n_free = 1


    # wsum = WeightedSum(nu_max=nu_max, n_free=n_free)
    # print(wsum(u=u))
    # print(wsum.all_tensors)
    # print(wsum.weights)

    block = CartesianEquivariantBasisBlock(nu_max=nu_max, n_free_max=2, in_dim=in_dim)

    print(block(u=u))

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

# cart_equiv = CartesianEquivariantBasisBlock(n=n, n_free_max=n_free_max, in_dim=in_dim)
# print(cart_equiv(tensors_in=torch.arange(1, n * in_dim + 1).reshape(n, in_dim)))

# contraction = CartesianContraction(n_free=1, nu=nu_max, in_dim=in_dim)
# print(contraction(tensors_in=tensors_in))
# print(tc.count_contractions(n=nu_max,k=k))

'''
