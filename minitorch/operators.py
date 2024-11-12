"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """Returns True if x is less than y. Otherwise, False."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Returns True if two numbers are equal. Otherwise, False."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    """Returns True if x is close to y within a tolerance. Otherwise, False."""
    return (x - y < tol) and (y - x < tol)


def sigmoid(x: float) -> float:
    """Applies the sigmoid function to a number."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the ReLU activation to a number."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Returns the natural logarithm of a number."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Returns the exponential of a number."""
    return math.exp(x)


def inv(x: float) -> float:
    """Returns the reciprocal of a number."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second arg."""
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second arg."""
    return d if x > 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element in an iterable.

    Args:
    ----
        fn: A function that takes a float and returns a float.

    Returns:
    -------
        A function that takes an iterable of floats and returns an iterable of floats with the function applied to each element.

    """

    def apply(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Applies a given function to combine elements from two iterables.

    Args:
    ----
        fn: A function that takes two floats and returns a float.

    Returns:
    -------
        A function that takes two iterables of floats and returns an iterable of floats with the function applied to each pair of elements.

    """

    def apply(list1: Iterable[float], list2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(list1, list2)]

    return apply


def reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], float], float]:
    """Reduces an iterable to a single value using a given function.

    Args:
    ----
        fn: A function that takes two floats and returns a float.

    Returns:
    -------
        A function that takes an iterable of floats and a starting float value, and returns a single float value obtained by applying the function cumulatively to the elements of the iterable.

    """

    def apply(ls: Iterable[float], start: float) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result

    return apply


def negList(list: Iterable[float]) -> Iterable[float]:
    """Negates each element in an iterable.

    Args:
    ----
        list: An iterable of floats.

    Returns:
    -------
        An iterable of floats where each element is negated.

    """
    neg_list = map(neg)
    return neg_list(list)


def addLists(list1: Iterable[float], list2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements from two iterables.

    Args:
    ----
        list1: The first iterable of floats.
        list2: The second iterable of floats.

    Returns:
    -------
        An iterable of floats where each element is the sum of the corresponding elements from the input iterables.

    """
    add_lists = zipWith(add)
    return add_lists(list1, list2)


def sum(list: Iterable[float]) -> float:
    """Calculates the sum of elements in an iterable.

    Args:
    ----
        list: An iterable of floats.

    Returns:
    -------
        The sum of the elements in the iterable.

    """
    sum_fn = reduce(add)
    return sum_fn(list, 0.0)


def prod(list: Iterable[float]) -> float:
    """Calculates the product of elements in an iterable.

    Args:
    ----
        list: An iterable of floats.

    Returns:
    -------
        The product of the elements in the iterable.

    """
    prod_fn = reduce(mul)
    return prod_fn(list, 1.0)
