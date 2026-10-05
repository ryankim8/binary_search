from src.argmin import bounded_argmin, find_boundaries, argmin
import math


def test__argmin_1():
    epsilon = 1.0
    lo = -20
    hi = 20
    x_min = 5
    f = lambda x: (x-x_min)**2
    assert abs(bounded_argmin(f,lo,hi,epsilon)-x_min) <= epsilon

def test__argmin_2():
    epsilon = 1e-3
    lo = -20
    hi = 20
    x_min = 5
    f = lambda x: (x-x_min)**2
    assert abs(bounded_argmin(f,lo,hi,epsilon)-x_min) <= epsilon

def test__argmin_3():
    epsilon = 1e-6
    lo = -20
    hi = 20
    x_min = 5
    f = lambda x: (x-x_min)**2
    assert abs(bounded_argmin(f,lo,hi,epsilon)-x_min) <= epsilon

def test__argmin_4():
    epsilon = 1e-9
    lo = -20
    hi = 20
    x_min = 5
    f = lambda x: (x-x_min)**2
    assert abs(bounded_argmin(f,lo,hi,epsilon)-x_min) <= epsilon

def test__argmin_5():
    epsilon = 1e-12
    lo = -20
    hi = 20
    x_min = 5
    f = lambda x: (x-x_min)**2
    assert abs(bounded_argmin(f,lo,hi,epsilon)-x_min) <= epsilon

def test__argmin_6():
    epsilon = 1e-6
    lo = -1e20
    hi = 1e20
    x_min = 5000
    f = lambda x: (x-x_min)**2
    assert abs(bounded_argmin(f,lo,hi,epsilon)-x_min) <= epsilon

def test__argmin_7():
    epsilon = 1e-6
    lo = -1e20
    hi = 0
    x_min = 5000
    f = lambda x: (x-x_min)**2
    assert abs(bounded_argmin(f,lo,hi,epsilon)-hi) <= epsilon

def test__argmin_8():
    epsilon = 1e-6
    lo = 0
    hi = 1e20
    x_min = 5000
    f = lambda x: (x-x_min)**2
    assert abs(bounded_argmin(f,lo,hi,epsilon)-x_min) <= epsilon

def test__argmin_9():
    epsilon = 1e-6
    lo = 0
    hi = 1e20
    x_min = -5000
    f = lambda x: (x-x_min)**2
    assert abs(bounded_argmin(f,lo,hi,epsilon)-lo) <= epsilon

def test__argmin_10():
    epsilon = 1e-6
    lo = 0
    hi = 1e20
    x_min = -5000
    f = lambda x: x
    assert abs(bounded_argmin(f,lo,hi,epsilon)-lo) <= epsilon


def test__find_boundaries_1():
    x_min = 0
    f = lambda x: (x-x_min)**2
    lo,hi = find_boundaries(f)
    assert lo <= x_min <= hi

def test__find_boundaries_2():
    x_min = 10
    f = lambda x: (x-x_min)**2
    lo,hi = find_boundaries(f)
    assert lo <= x_min <= hi

def test__find_boundaries_3():
    x_min = -10
    f = lambda x: (x-x_min)**2
    lo,hi = find_boundaries(f)
    assert lo <= x_min <= hi

def test__find_boundaries_4():
    x_min = 1e10
    f = lambda x: (x-x_min)**2
    lo,hi = find_boundaries(f)
    assert lo <= x_min <= hi

def test__find_boundaries_5():
    x_min = -1e10
    f = lambda x: (x-x_min)**2
    lo,hi = find_boundaries(f)
    assert lo <= x_min <= hi

def test__argmin_simple_1():
    epsilon = 1e-3
    x_min = 0
    f = lambda x: (x-x_min)**2
    assert abs(argmin(f,epsilon)-x_min) <= epsilon

def test__argmin_simple_2():
    epsilon = 1e-3
    x_min = 10
    f = lambda x: (x-x_min)**2
    assert abs(argmin(f,epsilon)-x_min) <= epsilon

def test__argmin_simple_3():
    epsilon = 1e-3
    x_min = -10
    f = lambda x: (x-x_min)**2
    assert abs(argmin(f,epsilon)-x_min) <= epsilon

def test__argmin_simple_4():
    epsilon = 1e-3
    x_min = -1e10
    f = lambda x: (x-x_min)**2
    assert abs(argmin(f,epsilon)-x_min) <= epsilon

def test__argmin_simple_5():
    epsilon = 1e-3
    x_min = 1e10
    f = lambda x: (x-x_min)**2
    assert abs(argmin(f,epsilon)-x_min) <= epsilon


# The functions below are all convex, but their minima are hard to locate
# using "algebraic" techniques:
#   * abs, sqrt((x-a)**2+delta**2), max, and x**2 + c*abs(x-a) are not
#     differentiable at the minimum, so there is no derivative to set
#     equal to zero there;
#   * cosh and logsumexp are transcendental, so f'(x) == 0 can only be
#     solved by inverting transcendental functions;
#   * x**2 + exp(x) is transcendental with no closed form solution for
#     its minimum at all;
#   * the max functions are piecewise defined, so which formula applies
#     depends on the unknown location of the minimum.
# Binary search is oblivious to all of this: it never differentiates, never
# manipulates the functions symbolically, and only ever compares two values
# of f, so it handles every one of these functions in the same way.


def test__argmin_quartic():
    # (x-x_min)**4 is very flat near its minimum, so f's value is a poor
    # proxy for the distance from x_min
    x_min = 7
    epsilon = 1e-6
    f = lambda x: (x - x_min)**4
    assert abs(argmin(f, epsilon) - x_min) <= epsilon


def test__argmin_abs():
    # the minimum of abs(x-x_min) is at its kink
    x_min = -4
    epsilon = 1e-6
    f = lambda x: abs(x - x_min)
    assert abs(argmin(f, epsilon) - x_min) <= epsilon


def test__argmin_sqrt_smoothed_abs():
    # a differentiable approximation of abs, still minimized at x_min
    x_min = 2
    delta = 1e-3
    epsilon = 1e-6
    f = lambda x: math.sqrt((x - x_min)**2 + delta**2)
    assert abs(argmin(f, epsilon) - x_min) <= epsilon


def test__argmin_max_of_lines():
    # max of two lines with opposite slopes: the kink is the minimum
    x_min = 2
    epsilon = 1e-6
    f = lambda x: max(0.5*x + 1, -x + 4)
    assert abs(argmin(f, epsilon) - x_min) <= epsilon


def test__argmin_cosh():
    x_min = 0
    epsilon = 1e-6
    f = lambda x: math.cosh(x - x_min)
    assert abs(argmin(f, epsilon) - x_min) <= epsilon


def test__argmin_cosh_shifted():
    x_min = 3
    epsilon = 1e-6
    f = lambda x: math.cosh(x - x_min)
    assert abs(argmin(f, epsilon) - x_min) <= epsilon


def test__argmin_logsumexp():
    # a smooth version of 2*abs(x-x_min)
    x_min = 1
    epsilon = 1e-6
    f = lambda x: math.log(math.exp(x - x_min) + math.exp(x_min - x))
    assert abs(argmin(f, epsilon) - x_min) <= epsilon


def test__argmin_quadratic_plus_abs():
    # the kink of abs moves the minimum away from the minimum of x**2;
    # the two cases of the definition give x_min == c/2 here
    a, c = 5.0, 1.0
    x_min = c/2
    epsilon = 1e-6
    f = lambda x: x**2 + c*abs(x - a)
    assert abs(argmin(f, epsilon) - x_min) <= epsilon


def test__argmin_max_of_quadratics():
    # the minimum is at the crossing of the two parabolas
    a, b = -3, 7
    x_min = (a + b)/2
    epsilon = 1e-6
    f = lambda x: max((x - a)**2, (x - b)**2)
    assert abs(argmin(f, epsilon) - x_min) <= epsilon


def test__argmin_weighted_sum_of_quadratics():
    # the minimum is the weighted average of the a_i
    data = [(-5, 1), (0, 2), (10, 3)]
    x_min = sum(w*a for a, w in data) / sum(w for a, w in data)
    epsilon = 1e-6
    f = lambda x: sum(w*(x - a)**2 for a, w in data)
    assert abs(argmin(f, epsilon) - x_min) <= epsilon


def test__argmin_no_closed_form_minimizer():
    # f'(x) == 2*x + exp(x) == 0 has no closed form solution,
    # so there is no x_min to compare against;
    # instead, check that argmin beats a grid of candidate minimizers
    f = lambda x: x**2 + math.exp(x)
    xs = [i/20 for i in range(-20, 21)]
    x_star = argmin(f, 1e-4)
    assert f(x_star) < min(f(x) for x in xs)
