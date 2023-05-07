import itertools
import string
import torch

from typing import Optional, Tuple, List, Generator
from math import factorial as fact
from scipy.stats import ortho_group

def pick_pairs(n_indices: int, n_contractions: int) -> Tuple[Tuple[str, str]]:
    """
    gives the combinations that `n_contractions` pairs of indices from a tensor with n_indices

    e.g. if n_indices = 4 and n_contractions = 2
    -> (('i','j'), ('k', 'l')), (('i','k'), ('j', 'l')), (('i','l'), ('j', 'k'))

    need to comment the code here **

    Parameters:
          n_indices: number of free indices before contraction
          n_contractions:

    Returns:
          comb: all possible combinations as a series of indices starting at 'i'
    """
    indices = [char for char in string.ascii_lowercase[8:]]
    n_contractions = int(n_contractions)

    points = list(indices[:n_indices])
    pairs = list(itertools.combinations(points, 2))
    for comb in itertools.combinations(pairs, n_contractions):
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


def count_contractions(n: int, n_contractions: int) -> int:
    """
    gives the number of different ways n indices can have n_contractions over pairs of indices


    Parameters:
          n: number of free indices before contraction
          n_contractions: number of pairs of indices to contract over

    Returns:
          n_combinations: total number of combinations for th
    """
    if n_contractions > n / 2 or n_contractions < 0:
        return 0

    else:
        return int(
            fact(n)
            / (
                fact(n - 2 * n_contractions)
                * fact(n_contractions)
                * 2**n_contractions
            )
        )


# %%
def count_contraction_paths(nu_max: int, c_out: int) -> int:
    """
    gives the total number of ways that you will be ables to produce a tensor
    of rank `c_out` all the possible routes from TPs of 1 to `nu_max`

    will probably have to revisit when I allow different inputs

    Parameters:
          nu_max: maximum tensor product order
          c_out: number of free indices of final equivariants

    Returns:
          tot: total number of paths to produce this
    """
    tot = 0

    for nu in range(1, nu_max + 1):
        k = (nu * 1 - c_out) / 2

        # so that only when k is integer do we consider it
        # can't have half a contraction!
        if k.is_integer():
            k = int(k)
            tot += count_contractions(n=nu, k=k)

    return tot


def cons_to_einsum(
    cons: Tuple[Tuple[str, str]],
    n: int,
    split: Optional[List[int]] = None,
    indices: Optional[List[str]] = None,
    extra_dims: Optional[int] = 0,
) -> str:
    """
    Generates the Einstein summation (einsum) subscript string for a set of contractions given in order.

    This function takes a set of contractions and generates the corresponding einsum subscript string that can be used
    with the `torch.einsum` function.

    For example the args cons=[('j','k')], n=4, split=[0,1,2,1,0] return 'aI,ai,ajj,al,aJ->il'

    Upper-case indices represent scalars for clarity and should only ever appear as an input (left of '->')

    Parameters:
        cons (Tuple[Tuple[str, str]]): A tuple of tuples representing the indices that will be contracted.
        n (int): The total number of indices, i.e., tensor rank of input.
        split (Optional[List[int]]): An optional list of integer sizes to group indices.
        indices (Optional[List[str]]): An optional list of custom index labels.
        extra_dims (Optional[bool]): The number of extra dimensions from channels and neighbourhood
        *** this can probably get restored to multi_channel when scatter is added later on

    Returns:
        str: The einsum subscript string for `torch.einsum`.
    """

    assert n == sum(split)
    assert len(cons) <= n / 2

    if not indices:
        indices = [char for char in string.ascii_lowercase[8:]]

    rem_letters = "".join(indices[:n])
    einsum_indices = "".join(indices[:n])

    for con in cons:
        einsum_indices = einsum_indices.replace(con[1], con[0])

        rem_letters = rem_letters.replace(con[0], "")
        rem_letters = rem_letters.replace(con[1], "")

    multi_channel = True

    scalar_indices = [char for char in string.ascii_uppercase[8:]]

    start = 0

    einsum = []

    for rank in split:
        # scalars
        if rank == 0:
            einsum.append(scalar_indices[0])
            scalar_indices.pop(0)

        else:
            einsum.append(einsum_indices[start : start + rank])

            start += rank

    if extra_dims:  # instead of ellipsis for safety too
        # what is this for? when we've got neighbours?
        # einsum = [string.ascii_lowercase[:extra_dims] + letters for letters in einsum]
        einsum = ["..." + letters for letters in einsum]

    einsum = ",".join(einsum)

    einsum += "->"
    einsum += "..." + rem_letters

    return einsum


def compute_cons_all_combs(
    n: int, k: int, tensors_in: torch.Tensor, indices: list[str]
) -> list[torch.Tensor]:
    """
    calls `compute_cons` for all combinations of the contractions for a given number of connections

    Parameters:
        n: number of indices (should be able to compute implicitly)
        tensors_in: list of input tensors
        indices: how we label our indices (default is i -> z)


    Returns:
          complete_con_prods: list of tensors split up by tensor rank
    """

    complete_con_prods = []

    # find k
    cons_combs = list(pick_pairs(n=n, k=k, indices=indices))

    for cons in cons_combs:
        einsum = cons_to_einsum(cons=cons, n=n, indices=indices)
        complete_con_prods.append(torch.einsum(einsum, *tensors_in))

    return complete_con_prods


def find_combinations(n, nu, c_max: Optional[int] = 2):
    def helper(n, nu, current_combination, start):
        if n == 0 and nu == 0:
            combinations.append(current_combination[:])
            return

        if n < 0 or nu == 0:
            return

        for i in range(start, c_max + 1):
            current_combination.append(i)
            helper(n - i, nu - 1, current_combination, i)
            current_combination.pop()

    combinations = []
    helper(n, nu, [], 0)
    return combinations


def tensors_to_n(
    n: int, max_tensor_out_rank: int, tensors_in: list[torch.Tensor]
) -> list[list[torch.Tensor]]:
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
        k = int(
            n / 2 - i
        )  # should do something here to safeguard against something going wrong

        if k < 0:  # need to throw an appropriate error here
            pass

        tensors_out.append(
            compute_cons_all_combs(n=n, k=k, tensors_in=tensors_in, indices=indices)
        )

    return tensors_out


def tensor_prod(r: torch.Tensor, order: int) -> torch.Tensor:
    """
    ***
    change order -> rank
    change name to something more descriptive of 'self' TP aspect

    creates self tensor products of displacement vector r of rank order
    i.e. r \otimes ... order times ... \otimes r = r^{\otimes order}

    the non-trivial part of this function is creating an einsum expression
    e.g. for order = 2, einsum = 'i...,j...->...ij'

    in the usual use-case, r is the normalised direction to the central node
    from a neighbour.

    This function is vectorised, taking
    in shape: dim x n_neighbours
    out shape: n_neighbours x dim^c_out
    could perhaps change this for the ordering **

    Parameters:
        r: vectors to tensor product with itself
        order: the number of times to do the tensor product

    Returns:
          r_tensor: output tensor of rank-order
    """

    # clean this part up **
    indices = [char for char in string.ascii_lowercase[8 : 8 + order]]  # i -> z
    einsum_start = ",".join([f"...{index}" for index in indices])
    joined_indices = "".join(indices)
    einsum_end = f"...{joined_indices}"
    einsum = einsum_start + "->" + einsum_end

    # what is r^\otimes 0? **
    if order == 0:
        return torch.ones(r.shape[0], 1)

    # using einsum to carry out the tp
    r_tensor = torch.einsum(
        einsum,
        [
            r,
        ]
        * order,
    )

    return r_tensor


def e_rbf(r: torch.Tensor, n: int, c: Optional[int] = 2) -> torch.Tensor:
    """
     Orthogonal set of radial basis functions used in the DimeNet paper (https://arxiv.org/pdf/2003.03123.pdf)

    Parameters:
        r: absolute distance from
        n: upto order n basis functions
        c: parameter

    Returns:
          e_rbf: the value of the n radial basis functions at distance r
    """

    e_rbf_tilde = (2 / c) ** 0.5 * torch.sin(n * r) / r

    # add polynomial back in later
    u = 1  # - (p+1)*(p+2)/2 * r**p + p*(p+2)*r**(p+1) - p*(p+1)/2**r**(p+2)

    return e_rbf_tilde * u


def contraction_is_valid(num_legs_in: int, num_legs_out: int) -> bool:
    """
    Tests validity of tensor contraction.
    Will become more complex as we add O(3) symmetries

    Parameters:
        n_free_indices_total: number of free indices BEFORE contractions
        n_indices_desired: desired number of free indices AFTER contractions

    Returns:
        validity: bool as to whether is valid contraction =
    """

    n_contractions = (num_legs_in - num_legs_out) / 2

    return n_contractions.is_integer() and n_contractions > -1


def create_zero_feature_tensors(
    feature_rank_max: int, n_nodes: int, n_channels: int, dim: Optional[int] = 3, populate: Optional[bool] = False) -> List[torch.Tensor]:
    """
    Creates a list of features tensors of zeros that get populated throughout the layers.

    Currently has separate tensors for central and neighbouring nodes (removed once I graph-ify things!).

    The shape of `h` is a list of len(h): `feature_rank_max + 1` (+1 as we include scalars of rank 0 too).
    within this list, each tensor is n_nodes x n_channels x tensor_shape

    tensor_shape = (dim,) * rank

    h_central is the same apart from no n_nodes dimension

    Parameters:
          feature_rank_max: the max rank of feature vectors that we keep
          n_channels: number of channels
          dim: dimension of space

    Returns:
        h: neighbouring nodes
        h_central: central node
    """
    h = []

    for feature_rank in range(0, feature_rank_max + 1):
        if feature_rank == 0:
            tensor_shape = (
                n_nodes,
                n_channels,
            ) + (1,)

        else:
            tensor_shape = (
                n_nodes,
                n_channels,
            ) + (dim,) * feature_rank

        h.append(torch.zeros(tensor_shape))

    if populate:
        h = [torch.nn.init.normal_(g) for g in h]

    return h


def update_feature_tensors(
    channel_weights: List[torch.Tensor],
    message_weights: List[torch.Tensor],
    h_central: List[torch.Tensor],
    messages: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Think about vectorising this function (although we may not need to give how pyg works)

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
    for c_out in range(feature_rank_max + 1):
        # Mix node states from previous layers
        h_central[c_out] = torch.einsum(
            "ij,j...->i...", channel_weights[c_out], h_central[c_out]
        )
        # Mix all messages of matching rank
        h_central[c_out] += torch.einsum(
            "ij,j...->i...", message_weights[c_out], messages[c_out]
        )

    return h_central

def linearise_features(h: List[torch.Tensor]) -> torch.Tensor:
    # i from 0 to feature_rank_max e.g. for feature_rank_max = 2
    # in: list with h[i] = [ [n_nodes, n_channels, 1], [n_nodes, n_channels, 2], [n_nodes, n_channels, 2, 2] ]
    # out: [n_nodes, (1 * n_channels) + (2 * n_channels) + (2 * 2 * n_channels)]
    # we do this such that we have n_channel lots of each feature in order then the next

    h_flattened = []

    # essentially need to flatten all the dimensions
    for h_i in h:

        h_flattened.append(
            h_i.flatten(start_dim=1, end_dim=-1)
        )

    return torch.cat(h_flattened, dim=1)

def init_orthogonal_weights(n_channels: int, extra_dim: int) -> torch.Tensor:

    return torch.stack([torch.from_numpy(ortho_group.rvs(n_channels)).to(torch.float32) for _ in range(extra_dim)])
