from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to a set of Scalar variables."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivative is 1 for both inputs."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Log of a number."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative is 1/x."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Multiply two numbers."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Derivative with respect to each input is the other input."""
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Inverse of a number."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of inv is -1/x^2."""
        (a,) = ctx.saved_values
        f_prime = operators.inv_back
        return f_prime(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Negate a number."""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of neg is -1."""
        return -1.0 * d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Sigmoid of a number."""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of sigmoid is sigmoid * (1 - sigmoid)."""
        (a,) = ctx.saved_values
        return d_output * operators.sigmoid(a) * (1.0 - operators.sigmoid(a))


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """ReLU of a number."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of ReLU is 1 if input is positive, 0 otherwise."""
        (a,) = ctx.saved_values
        return d_output * operators.relu_back(a, 1.0)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Exponential of a number."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of exp is exp when input is positive, -exp otherwise."""
        (a,) = ctx.saved_values
        return d_output * operators.exp(a)


class Lt(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Less than function."""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Derivative of less than is 0."""
        return 0.0, 0.0


class Eq(ScalarFunction):
    """Equal function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Equal function."""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Derivative of equal is 0."""
        return 0.0, 0.0
