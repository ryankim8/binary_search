'''
All of the functions in this file are classic leetcode style interview questions.
They all take a container xs as input and run in time O(log n),
where n is the length of xs.

JOKE: There are 2 hard problems in computer science:
1. cache invalidation,
2. naming things, and
3. off-by-1 errors.

It's really easy to have off-by-1 errors in these problems.
Pay very close attention to your list indexes and your < vs <= operators.
'''


def find_smallest_positive(xs):
    '''
    Assume that xs is a list of numbers sorted from LOWEST to HIGHEST.
    Find the index of the smallest positive number.
    If no such index exists, return `None`.

    HINT:
    This is essentially the binary search algorithm from class,
    but you're always searching for 0.

    >>> find_smallest_positive([-3, -2, -1, 0, 1, 2, 3])
    4
    >>> find_smallest_positive([1, 2, 3])
    0
    >>> find_smallest_positive([-3, -2, -1]) is None
    True
    '''
    def helper(lo, hi):
        if lo == hi:
            return lo
        mid = (lo + hi) // 2
        if xs[mid] <= 0:
            return helper(mid + 1, hi)
        else:
            return helper(lo, mid)
    x = helper(0, len(xs))
    return x if x < len(xs) else None

def find_largest_negative(xs, lo=0, hi=None):
    '''
    Assume that xs is a list of numbers sorted from LOWEST to HIGHEST.
    Find the index of the largest negative number.
    If no such index exists, return `None`.

    HINT:
    This is the mirror image of find_smallest_positive:
    both functions search for the boundary at 0,
    but they return different sides of that boundary.

    >>> find_largest_negative([-3, -2, -1, 0, 1, 2, 3])
    2
    >>> find_largest_negative([1, 2, 3]) is None
    True
    >>> find_largest_negative([-3, -2, -1])
    2
    '''
    def helper(lo, hi):
        if lo == hi:
            return lo
        mid = (lo + hi) // 2
        if xs[mid] >= 0:
            return helper(lo, mid)
        else:
            return helper(mid + 1, hi)
    x = helper(0, len(xs))
    return x - 1 if x > 0 else None


def find_smallest(xs, lo=0, hi=None):
    '''
    Assume that xs is a list of numbers that is strictly decreasing
    and then strictly increasing,
    so that xs has a unique smallest element.
    Return the index of that element, or `None` if xs is empty.

    NOTE:
    This is the discrete analogue of argmin in src/argmin.py:
    argmin minimizes a convex function over the reals,
    and find_smallest minimizes a list of numbers.

    >>> find_smallest([4, 3, 2, 1, 2, 3])
    3
    >>> find_smallest([1, 2, 3])
    0
    >>> find_smallest([3, 2, 1])
    2
    >>> find_smallest([]) is None
    True
    '''
    if len(xs) == 0:
        return None
    def helper(lo, hi):
        if lo == hi:
            return lo
        mid = (lo + hi) // 2
        if xs[mid] > xs[mid + 1]:
            return helper(mid + 1, hi)
        else:
            return helper(lo, mid)
    return helper(0, len(xs) - 1)


def count_repeats(xs, x):
    '''
    Assume that xs is a list of numbers sorted from HIGHEST to LOWEST,
    and that x is a number.
    Calculate the number of times that x occurs in xs.

    HINT:
    Use the following three step procedure:
        1) use binary search to find the lowest index with a value >= x
        2) use binary search to find the lowest index with a value < x
        3) return the difference between step 1 and 2
    I highly recommend creating stand-alone functions for steps 1 and 2,
    and write your own doctests for these functions.
    Then, once you're sure these functions work independently,
    completing step 3 will be easy.

    >>> count_repeats([5, 4, 3, 3, 3, 3, 3, 3, 3, 2, 1], 3)
    7
    >>> count_repeats([3, 2, 1], 4)
    0
    '''
    def bottomX(xs, x):
        def helper(lo, hi):
            if lo == hi:
                return lo
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                return helper(lo, mid)
            else:
                return helper(mid + 1, hi)
        return helper(0, len(xs))

    def topX(xs, x):
        def helper(lo, hi):
            if lo == hi:
                return lo
            mid = (lo + hi) // 2
            if xs[mid] < x:
                return helper(lo, mid)
            else:
                return helper(mid + 1, hi)
        return helper(0, len(xs))

    return topX(xs, x) - bottomX(xs, x)

