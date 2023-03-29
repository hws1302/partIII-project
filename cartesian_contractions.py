import numpy as np

import itertools
import string
import torch

from math import factorial as fact
def pick_pairs(n: int, k: int, numbers: bool = False): # what is the output of this func

    if numbers:
        torch.arange(30)

    else:
        indices = [char for char in string.ascii_lowercase[8:]]

    points = list(indices[:n])
    pairs = list(itertools.combinations(points, 2))
    for comb in itertools.combinations(pairs, k):
        used_points = set()
        valid = True
        for pair in comb:
            if pair[0] in used_points or pair[1] in used_points:
                valid = False
                break
            used_points.add(pair[0])
            used_points.add(pair[1])
        if valid:
            yield comb


def count_contractions(n: int, k: int) -> int:
    if k > n / 2:
        return 0

    else:
        return int(fact(n) / (fact(n - 2 * k) * fact(k) * 2 ** k))

#%%
def count_contraction_paths(nu_max: int, n_free: int) -> int:
    """"
    gives the total number of ways that you will be ables to produce a tensor
    of rank `n_free` all the possible routes from TPs of 1 to `nu_max`

    will probably have to revisit when I allow different inputs

    Parameters:
          nu_max: maximum tensor product order
          n_free: number of free indices of final equivariants

    Returns:
          tot: total number of paths to produce this
    """
    tot = 0

    for nu in range(1, nu_max + 1):

        k = (nu * 1 - n_free)/2

        # so that only when k is integer do we consider it
        # can't have half a contraction!
        if k.is_integer():

            k = int(k)
            tot += count_contractions(n=nu, k=k)

    return tot

# def cons_to_einsums(cons: tuple[tuple], n: int) -> list[str]:
#     """
#     gives the string necessary for einsum for a set of contractions given in order
#     e.g. the contractions (i,j) then (k,l) returns ['iikl->kl', 'kl->']
#
#     Parameters:
#           cons: the indices that will be contracted
#           n: total number of indices i.e. tensor rank
#
#     Returns:
#           einsums: list of einsum subscript strings for `torch.einsum`
#     """
#
#     einsums = []
#
#     for i, con in enumerate(cons):
#
#         # input given as n vectors, this special case taken away if we use `torch.outer`
#         # not sure if using `torch.outer` would defeat the point of what we are doing?
#         # will leave in this hacky solution until I figure it out
#         # also need to come up with nicer solution that allows for arbitrary number of indices
#         # current methods uses i -> z i.e. only 18
#         if i == 0:
#             rem_letters = ''.join(indices[:n])
#             einsum = ','.join(indices[:n])
#             # replace indices we are contracting on i.e. 'ij' changed to 'ii'
#             einsum = einsum.replace(con[1], con[0])
#
#         else:
#
#             rem_letters = rem_letters.replace(con[1], con[0])
#             einsum = rem_letters # need to choose which one to double
#
#         # remove letters that are used up in contraction
#         rem_letters = rem_letters.replace(con[0], '')
#         rem_letters = rem_letters.replace(con[1], '')
#
#         einsum += '->'
#         einsum += rem_letters
#         einsums.append(einsum)
#
#     return einsums

def cons_to_einsum(cons: tuple[tuple], n: int, indices: list[str] = None) -> str:
    """
    gives the string necessary for einsum for a set of contractions given in order
    e.g. the contractions (i,j), (k,l) returns 'iikk->'

    Parameters:
          cons: the indices that will be contracted
          n: total number of indices i.e. tensor rank

    Returns:
          einsum: einsum subscript string for `torch.einsum`
    """

    if not indices:
        indices = [char for char in string.ascii_lowercase[8:]]

    rem_letters = ''.join(indices[:n])
    einsum = ','.join(indices[:n])

    for con in cons:

        einsum = einsum.replace(con[1], con[0])

        rem_letters = rem_letters.replace(con[0], '')
        rem_letters = rem_letters.replace(con[1], '')

    einsum += '->'
    einsum += rem_letters

    return einsum

# def compute_cons(einsums: list[str], tensors_in: list[torch.Tensor]) -> torch.Tensor:
#     """
#     computes contractions iteratively on `tensor_in`
#
#     Parameters:
#           einsums: list of einsum subscript strings
#           tensors_in: total number of indices i.e. tensor rank
#
#     Returns:
#           complete_con_prod: resulting tensor after completing all contractions given by `einsums`
#     """
#
#     for i, einsum in enumerate(einsums):
#
#         if i == 0: # hacky solution that could be removed if used `torch.outer`
#             semi_con_prod  = torch.einsum(einsum, *tensors_in)
#
#         else:
#             semi_con_prod = torch.einsum(einsum, semi_con_prod)
#
#     complete_con_prod = semi_con_prod # just to be clear
#
#     return complete_con_prod

def compute_cons_all_combs(n: int, k: int, tensors_in: torch.Tensor, indices: list[str]) -> list[torch.Tensor]:
    """
    calls `compute_cons` for all combinations of the contractions for a given number of connections

    Parameters:
        n: number of indices (should be able to compute implicitly)
        tensors_in: list of input tensors
        indices: how we label our indices (default is i -> z)


    Returns:
          complete_con_prods: list of tensors split up by tensor rank
    """

    complete_con_prods= []

    # find k
    cons_combs = list(pick_pairs(n=n, k=k, indices=indices))

    for cons in cons_combs:

        einsum = cons_to_einsum(cons=cons, n=n, indices=indices)
        complete_con_prods.append(torch.einsum(einsum, *tensors_in))

    return complete_con_prods


def tensors_to_n(n: int, max_tensor_out_rank: int, tensors_in: list[torch.Tensor]) -> list[list[torch.Tensor]]:
    """
    creates all the possible tensors of each rank up to some `max_tensor_out_rank`

    Parameters:
        n: number of indices (should be able to compute implicitly)
        max_tensor_out_rank: maximum rank of the output tensors
        tensors_in: list of input tensors


    Returns:
          complete_con_prods: list of tensors split up by tensor rank
    """
    tensors_out = []
    indices = [char for char in string.ascii_lowercase[8:]]  # i -> z

    for i in range(max_tensor_out_rank):

        k = int(n/2 - i) # should do something here to safeguard against something going wrong

        if k < 0: # need to throw an appropriate error here
            pass

        tensors_out.append(compute_cons_all_combs(n=n, k=k, tensors_in=tensors_in, indices=indices))

    return tensors_out

# just so it doesn't run when we import the module!
if __name__ == '__main__':

    print(list(pick_pairs(n=4, k=2, numbers=True)))

    # u = torch.arange(1,4)
    # v = torch.arange(4,7)
    # s = torch.arange(7,10)
    # t = torch.arange(10,13)
    # p = torch.arange(13,16)
    # q = torch.arange(16,19)
    #
    # indices = [char for char in string.ascii_lowercase[8:]] # i -> z
    #
    # # cons_combs = list(pick_pairs(n=4, k=1, indices=indices))
    # #
    # # cons = cons_to_einsums(cons=cons_combs[0], n=4)
    #
    # print(tensors_to_n(n=4, max_tensor_out_rank=2, tensors_in=[u,v,s,t]))
    #
    # dim = 30
    # I = np.random.rand(dim, dim, dim, dim)
    # C = np.random.rand(dim, dim)
    #
    # print(oe.contract_path('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C))
    #
    # np.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
    # oe.contract_path('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
