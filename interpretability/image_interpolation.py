from abc import (ABC, abstractmethod)
from typing import (Tuple, Optional)
from torch import Tensor
import torch


class AbstractImageInterpolator(ABC):
    """Abstract image interpolation."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def update_parameters(self, *args):
        pass

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass


class LinearImageInterpolator(AbstractImageInterpolator):

    def __init__(self,
                 num_steps: int,
                 ) -> None:
        """Linear interpolator for blending an image between a baseline and the original image.

        Args:
            num_steps (int): The number of interpolation steps.
        """
        super().__init__()
        self.num_steps = num_steps
        self.baseline = None  # type: Optional[Tensor]
        self.alphas = None  # type: Optional[Tensor]
        self.expanded_alphas = None  # type: Optional[Tensor]
        self.expanded_baseline = None  # type: Optional[Tensor]

    def update_parameters(self,
                          baseline: Tensor
                          ) -> None:
        """Updates the parameters (baseline) to prepare for computation.

        Args:
            baseline (Tensor): The baseline used for the blending.

        Returns:
            None
        """
        self.baseline = baseline
        self.alphas = torch.linspace(0, 1, self.num_steps + 1, device=baseline.device)
        self.expanded_alphas = self.alphas.view(-1, *baseline.ndim * (1,))
        self.expanded_baseline = baseline.unsqueeze(0).expand(self.expanded_alphas.shape[0], *baseline.shape)

    def _validate_baseline_and_alphas(self):
        assert all((self.baseline is not None,
                    self.alphas is not None,
                    self.expanded_alphas is not None,
                    self.expanded_baseline is not None)), f'The parameters need to be updated before execution!'

    def execute(self,
                image: Tensor
                ) -> Tuple[Tensor, Tensor]:
        """Performs the execution of the blending.

        Args:
            image (Tensor): The image tensor which is blended with the baseline.

        Returns:
            Tensor: A tensor which a linear blending between the baseline and the image.
        """
        self._validate_baseline_and_alphas()

        # image.requires_grad_(False)a
        difference = (image - self.baseline).unsqueeze(0).expand(self.alphas.shape[0], *image.shape)
        blended_images = self.expanded_baseline + self.expanded_alphas * difference, self.alphas

        blended_images[0].requires_grad_(True)
        return blended_images
