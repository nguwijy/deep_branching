import math
import sympy
import collections
import time
from operator import mul
from functools import reduce
from itertools import product


# ordered_partitions...
def partitions_with_count(k):
    """
    return all possible partitions of a set with n elements grouped by the number of elements in each partition

    for example, paritions_with_count(3) returns {(1, 2): 3, (3,): 1, (1, 1, 1): 1}
    (1, 2)      -> [1],[2,3];  [2],[1,3];  [3],[1,2]    -> total of 3 ways
    (3,)        -> [1,2,3]                              -> total of 1 way
    (1, 1, 1)   -> [1],[2],[3]                          -> total of 1 way
    """
    if len(k) == 1 and k[0] == 1:
        return {((1,),): 1}  # the only way to partition set with one element

    unordered_len = []
    for i in sympy.utilities.iterables.multiset_partitions(
            [(group, j) for group, tot in enumerate(k) for j in range(tot)]):
        count_len = []
        for j in i:
            # print(j)
            ans = [0] * len(k)
            for kk in j:
                ans[kk[0]] += 1
            count_len.append(tuple(ans))
        unordered_len.append(tuple(sorted(count_len)))
    # all_partitions = sympy.utilities.iterables.multiset_partitions(range(k[0]))
    # all_partitions = sympy.utilities.iterables.multiset_partitions([(group, j) for group, tot in enumerate(k) for j in range(tot)])
    # unordered_len = [tuple(sorted([len(j) for j in i])) for i in all_partitions]
    return collections.Counter(unordered_len)


def integer_partitions(target, nb_ele):
    """
    return all possible ways to sum to target using nb_ele, allowing 0 to be one of the element

    for example, integer_partitions(4, 2) returns
    ([0,4], [4,0], [1,3], [3,1], [2,2])
    """
    if target == 0:
        return [[0] * nb_ele]

    ans = []
    for i in range(nb_ele):
        for part in sympy.utilities.iterables.ordered_partitions(target, m=i+1):
            arr = [0] * (nb_ele - (i + 1)) + part
            for j in sympy.utilities.iterables.multiset_permutations(arr):
                ans.append(j)
    return ans

def fdb_1d(k):
    """
    faa di bruno formula for 1 dimension

    for example, fdb_1d(k) return a list of size m x 3 such that
    d^k f(g(x)) = sum_{i=0}^{m-1} ans[i].coeff x f[ans[i].lamb]
                    x prod_{j=0}^{ans[i].s - 1} g[ans[i].l[j]]^[ans[i].k[j]],
    where f[n] means differentiate f n times
    ans[i].coeff is the index 0 of ans[i]
    ans[i].lamb is the index 1 of ans[i]
    ans[i].l is the key of the dictionary at index 2 of ans[i]
    ans[i].k is the value of the dictionary at index 2 of ans[i]
    ans[i].s is the size of the dictionary at index 2 of ans[i]
    """
    fdb = collections.namedtuple('fdb', 'coeff lamb l_and_k')
    return [fdb(count, len(par), collections.Counter(par))
            for par, count in partitions_with_count(k).items()]


def fdb_nd(n, ks):
    """
    the main function, multivariate faa di bruno formula for
    d^k1/dx1 d^k2/dx2 ... d^kd/dxd f(g1(x), g2(x), ..., gn(x))

    this implementation is derived from the R package kStatistics::MFB
    """
    fdb = collections.namedtuple('fdb', 'coeff lamb l_and_k')
    ans = []

    for par in product(*[integer_partitions(k, n) for k in ks]):
        # transpose par without numpy
        par = list(zip(*par))
        # k! / (par0! par1! par2! ...)
        p = reduce(mul, [math.factorial(ele) for ele in ks]) // reduce(mul, [math.factorial(ele) for group in par for ele in group])
        for cartesian in product(*[fdb_1d(ele) for ele in par]):
            coeff, lamb, l_and_k = p, [], {}
            for idx, ele in enumerate(cartesian):
                coeff *= ele.coeff
                lamb.append(ele.lamb)
                for l, k_arr in ele.l_and_k.items():
                    if l not in l_and_k:
                        # if the dictionary does not have the key of l, initialize one using zero list
                        l_and_k[l] = [0] * len(cartesian)
                    l_and_k[l][idx] = k_arr
            ans.append(fdb(coeff, tuple(lamb), l_and_k))
    return ans


if __name__ == "__main__":
    print(fdb_nd(2, (0, 0, 0)))
    # print(fdb_nd(3, (1, 2)))
    # print(len(fdb_nd(2, (1, 2))))
    # print(fdb_1d([1]))
    # print(fdb_1d(2))
    # print(fdb_1d([1, 2]))
    # print(list(integer_partitions(4, 2)))
    # print(partitions_with_count([1, 2]))
    # start = time.time()
    # print({k: fdb_nd(5, k) for k in range(1, 10)})
    # for k in range(1, 10):
    #     # print(fdb_nd(1, k))
    #     print(len(fdb_nd(1, k)))
    # print(f"Time taken: {time.time() - start} seconds.")
    # # print(len(fdb_nd(2,5)))
    # for i in integer_partitions(2, 3):
    #     print(i)
    # print(list(integer_partitions(5, 6)))
    # print(partitions_with_count(5))
