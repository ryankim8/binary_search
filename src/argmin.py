'''
A common data science problem is to find the minimum of an unknown function.
For example, all modern machine learning algorithms (including training large language models like chatgpt) are implemented this way.
In this file, you will explore some basic methods for finding the minimum using binary search.

The main function of this file is called argmin.
It takes another function as a parameter, which might feel unusual to you.
Consider the example quadratic function below:

>>> def f(x):
...    return (x-5)**2

The minimum of `f` is 5, and f(5) = 0.
In calculus, we compute the minimum of this function by taking the derivative f'
and setting it to 0.

The argmin function will do this work for us automatically:

>>> int(argmin(f))
5

Notice that when we call the argmin function, we pass `f` and not `f()`;
that is, we are passing the function itself and not calling the function.

It is often awkward to define simple functions using the `def` syntax,
and python has a shorter `lambda` syntax for defining 1-line functions.
The code below is equivalent to the code above:

>>> int(argmin(lambda x: (x-5)**2))
5

These "lambda functions" are also called "anonymous functions"
because they do not have a name.
The test cases make extensive use of these anonymous functions.

Here are some more examples using slightly more complex functions:

>>> int(argmin(lambda x: abs(x - 5)))
5

>>> g = lambda x: (x - 5)**2 if x > 5 else 3*(x - 5)**2
>>> int(argmin(g))
5

The pytest test cases have examples of much more complex functions
that would be impossible to find the minimum of using any standard
calculus or algebra tricks.

NOTE:
The argmin function works over floating point values,
but floating point values are hard to write tests for.
So the doctests above convert the results to ints for convenience.
'''


def argmin(f, epsilon=1e-3):
    '''
    Returns a number that is within epsilon of the value that minimizes f(x).

    NOTE:
    There is nothing to implement for this function.
    If you implement the find_boundaries and bounded_argmin functions correctly,
    then this function will work correctly too.
    '''
    lo, hi = find_boundaries(f)
    return bounded_argmin(f, lo, hi, epsilon)


def bounded_argmin(f, lo, hi, epsilon=1e-3):
    '''
    Assumes that f is an input function that takes a float as input and returns a float with a unique global minimum,
    and that lo and hi are both floats satisfying lo < hi.
    Returns a number that is within epsilon of the value that minimizes f(x) over the interval [lo,hi]

    HINT:
    The basic algorithm is:
        1) The base case is when hi-lo < epsilon
        2) For each recursive call:
            a) select two points m1 and m2 that are between lo and hi
            b) one of the 4 points (lo,m1,m2,hi) must be the smallest;
               depending on which one is the smallest,
               you recursively call your function on the interval [lo,m2] or [m1,hi]

    NOTE:
    The runtime of this algorithm is O(log(1/epsilon)).
    Notice that there is no "n" / size parameter here at all.

    In this class, you will not be responsible for detailed proofs of runtimes like this.
    But here is an example proof of this runtime:

        **Proof**
        Each recursive call shrinks the interval to length <= (3/3)*(hi-lo).
        After k calls, length <= (2/3)^k * (hi-lo).
        We stop when length < epsilon, so we set

            (2/3)^k * (hi-lo) < epsilon

        then solve for k to get

            k > log_(2/3) (epsilon / (hi-lo))

        since log has base less than 1,
        the inequality flips when we take it of both sides.
        Negating the base to put the log above 1 gives us

            k > log(3/2) ((hi-lo) / epsilon)

        then applying big-O notation,
        the base of the log and the (hi-lo) factor drop out,
        giving us k = O(log 1/epsilon).
    '''
    if hi - lo < epsilon:
        return (hi + lo) / 2
    m1 = lo + (hi - lo) * .5
    m2 = lo + (hi - lo) * .51
    if f(m1) < f(m2):
        return bounded_argmin(f, lo, m2, epsilon)
    else:
        return bounded_argmin(f, m1, hi, epsilon)


def find_boundaries(f):
    '''
    Returns a tuple (lo,hi).
    If f is a convex function, then the minimum is guaranteed to be between lo and hi.
    This function is useful for initializing argmin.

    HINT:
    Begin with initial values lo=-1, hi=1.
    Let mid = (lo+hi)/2
    if f(lo) > f(mid):
        recurse with lo*=2
    elif f(hi) < f(mid):
        recurse with hi*=2
    else:
        you're done; return lo,hi
    '''
    def helper(lo, hi):
        mid = (lo + hi) / 2
        if f(lo) < f(mid):
            return helper(lo * 2, hi)
        elif f(hi) < f(mid):
            return helper(lo, hi * 2)
        else:
            return (lo, hi)
    return helper(-1, 1)
