import math
from fractions import Fraction
import numpy as np
from scipy.special import loggamma


def pochhammer(a, n):
    if n == 0:
        return Fraction(1, 1)
    result = Fraction(1, 1)
    for i in range(n):
        result *= a + i
    return result


def krawtchouk_poly(n, x, sample_count, p):
    """
    Paper hypergeometric definition:
    K_n(x; p, N) = 2F1(-n, -x, -N, 1/p), x=0..N.

    Here sample_count is the number of grid samples, so N = sample_count - 1.
    """
    N = sample_count - 1
    a = Fraction(-n, 1)
    b = Fraction(-x, 1)
    c = Fraction(-N, 1)
    z = Fraction(1, 1) / Fraction(p)

    value = Fraction(0, 1)
    for k in range(n + 1):
        num = pochhammer(a, k) * pochhammer(b, k)
        den = pochhammer(c, k) * math.factorial(k)
        value += num / den * (z ** k)

    return float(value)


def log_weight(x, sample_count, p):
    """log w(x;p,N), with N = sample_count - 1 and x=0..N."""
    N = sample_count - 1
    return (
        loggamma(N + 1)
        - loggamma(x + 1)
        - loggamma(N - x + 1)
        + x * math.log(p)
        + (N - x) * math.log(1 - p)
    )


def rho(n, sample_count, p):
    """
    Paper normalization constant:
    rho(n;p,N) = (-1)^n n! / (-N)_n * ((1-p)/p)^n.

    Since (-N)_n = (-1)^n N!/(N-n)!, rho is positive and computed in log form.
    """
    if n == 0:
        return 1.0
    N = sample_count - 1
    log_rho = (
        n * math.log((1 - p) / p)
        + loggamma(n + 1)
        + loggamma(N - n + 1)
        - loggamma(N + 1)
    )
    return math.exp(log_rho)


def krawtchouk_normalized(n, x, sample_count, p):
    K = krawtchouk_poly(n, x, sample_count, p)
    w = math.exp(log_weight(x, sample_count, p))
    r = rho(n, sample_count, p)
    return K * math.sqrt(w / r)


def precompute_K(max_order, sample_count, p):
    if max_order > sample_count:
        raise ValueError(
            f"max_order ({max_order}) must be <= sample_count ({sample_count})"
        )

    K = np.zeros((max_order, sample_count), dtype=np.float64)
    for n in range(max_order):
        for x in range(sample_count):
            K[n, x] = krawtchouk_normalized(n, x, sample_count, p)
    return K
