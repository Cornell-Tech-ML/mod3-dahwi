from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # implement central difference
    return (
        f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1 :])
        - f(*vals[:arg], vals[arg] - epsilon, *vals[arg + 1 :])
    ) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for this variable.

        Args:
        ----
            x: The derivative value to accumulate.

        """

    @property
    def unique_id(self) -> int:
        """Return the unique id of this variable.

        Returns
        -------
            int: The unique id of this variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Check if this variable is a leaf node.

        Returns
        -------
            bool: True if this variable is a leaf node, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Check if this variable is a constant node.

        Returns
        -------
            bool: True if this variable is a constant node, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parent variables of this variable.

        Returns
        -------
            Iterable[Variable]: The parent variables of this variable.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for this variable.

        Args:
        ----
            d_output: The derivative of the output variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples containing parent variables and their derivatives.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    res = []
    visited = set()

    def visit(var: Variable) -> None:
        # check if the variable is constant or already visited
        if var.is_constant() or var.unique_id in visited:
            return
        if not var.is_leaf():
            # if not constant or visited, visit its parents
            for p in var.parents:
                visit(p)
        res.insert(0, var)
        visited.add(var.unique_id)

    visit(variable)
    return res


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.
    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    Args:
    ----
        variable: The right-most variable
        deriv: The derivative of the output variable that we want to propagate backward to the leaves.

    """
    # get the topological order of the computation graph
    order = topological_sort(variable)

    # store the derivative of the output variable
    varToDerivativeMap = {variable.unique_id: deriv}
    # propagate the derivative backward
    for var in order:
        if var.is_leaf():
            # if the variable is a leaf node, accumulate the derivative
            var.accumulate_derivative(varToDerivativeMap[var.unique_id])
        else:
            # if the variable is not a leaf node, accumulate the derivative to its parents
            for parent, grad in var.chain_rule(varToDerivativeMap[var.unique_id]):
                varToDerivativeMap[parent.unique_id] = (
                    varToDerivativeMap.get(parent.unique_id, 0) + grad
                )


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved values."""
        return self.saved_values
