from abc import (ABC, abstractmethod)
from typing import Tuple
import torch
from torch import Tensor


class AbstractBaselineGenerator(ABC):
    """Abstract baseline generator."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def execute(self,
                x: Tensor
                ) -> Tensor:
        pass


class SelectiveMinimaBaseline(AbstractBaselineGenerator):

    def __init__(self,
                 reduction_axes: Tuple[int] = (0, 2, 3)
                 ) -> None:
        """Generates a baseline tensor with the same size as the input :attr:`x` with a constant minimum value
        on the selected reductions axis indices.

        Args:
            reduction_axes (Tuple[int]): The indexes of the axes to compute the minimum over.
        """
        super().__init__()
        self.reduction_axes = reduction_axes

    def _validate_tensor_reduction(self,
                                   x: Tensor
                                   ) -> None:
        assert x.ndim >= len(self.reduction_axes), \
            f'The number of dimensions of the input tensor is smaller than the number of reduction axes!'
        assert x.ndim > max(self.reduction_axes), \
            f'The maximum reduction axis index ({max(self.reduction_axes)}) is larger ' \
            f'than the number of dimensions in the input tensor ({x.ndim})!'

    def execute(self,
                x: Tensor
                ) -> Tensor:
        """Generates a new tensor with constant minima values on the selected reduction axes.

        Args:
            x (Tensor): The tensor to reduce.

        Returns:
            Tensor: The tensor with constant values on the reduction axes.
        """

        self._validate_tensor_reduction(x)

        minima = torch.amin(x, dim=self.reduction_axes, keepdim=True)
        ones = torch.ones_like(x)

        baseline = ones * minima
        baseline.requires_grad = False

        return baseline


class GlobalMinimumBaseline(AbstractBaselineGenerator):

    def __init__(self) -> None:
        """Generates a baseline tensor with the same size as the input :attr:`x` and the global minimum of :attr:`x`."""
        super().__init__()

    def execute(self,
                x: Tensor
                ) -> Tensor:
        """Generates a new tensor with constant minimum value.

        Args:
            x (Tensor):  The tensor to reduce.

        Returns:
            Tensor: The tensor with the global minimum of the input tensor :attr:`x`.
        """
        minimum = torch.min(x)
        ones = torch.ones_like(x)#, requires_grad=False)

        baseline = ones * minimum
        # baseline.requires_grad = False

        return baseline
