from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Softmax2d, Module

from interpretability.baseline_generator import AbstractBaselineGenerator
from interpretability.image_interpolation import AbstractImageInterpolator


class AttributionAlgorithm(ABC):
    """Abstract attribution algorithm."""

    def __init__(self,
                 model: Module
                 ) -> None:
        super().__init__()
        self.model = model

    @abstractmethod
    def execute(self,
                image: Tensor,
                idx_label: int
                ) -> Tensor:
        pass



class IG_simple(AttributionAlgorithm):

    def __init__(self,
                 model: Module,
                 baseline_generator: AbstractBaselineGenerator,
                 image_interpolator: AbstractImageInterpolator,
                 verbose: bool = True
                 ) -> None:
        """Integrated Gradient attribution algorithm.

        Args:
            model (Module): The model for the gradient computation.
            spatial_iterator (AbstractSpatialIterator): The interpolator for the evaluation locations.
            baseline_generator (AbstractBaselineGenerator): The generator for the baseline.
            image_interpolator (AbstractImageInterpolator): The interpolator for the generation of a blending.
            verbose (bool): If true the algorithm reports the progress to stdout.
        """
        super().__init__(model)
        self.baseline_generator = baseline_generator
        self.image_interpolator = image_interpolator
        self.verbose = verbose
        # self.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.softmax_fn = Softmax2d()

    def _compute_gradients(self,
                           image: Tensor,
                           batch_idx: int,
                           channel_idx: int,
                           ) -> Tensor:
        """Computes the gradients of the specified output w.r.t. the input image.

        Args:
            image (Tensor): The input image to the model.
            batch_idx (int): The batch index specifying the output for gradient computation.
            channel_idx (int): The channel index / label index specifying the output for gradient computation.


        Returns:
            Tensor: The tensor with the gradients of a specified output entry w.r.t. the input image.
        """
        probs = self.softmax_fn(self.model(image))
        grads = torch.autograd.grad(probs[batch_idx, channel_idx, :, :], image,
                                    grad_outputs=torch.ones_like(probs[batch_idx, channel_idx, :, :]))[0].detach()
        return grads

    @staticmethod
    def _integral_approximation(gradients: Tensor,
                                alphas: Tensor,
                                ) -> Tensor:
        """Computes the integral approximation based on the Riemann trapezoidal approach.

        Args:
            gradients (Tensor): The gradients of multiple backward steps collected in one tensor.
            alphas (Tensor): The step size between the gradients.

        Returns:
            Tensor: The normalized integral approximation of the tensors.
        """
        grads = (gradients[:-1] + gradients[1:]) / 2.0
        return torch.mean(grads, dim=0)
        # integrated_gradients = torch.zeros((gradients.shape[1:]), device=gradients.device)
        # for i in range(gradients.shape[0] - 1):
        #     step_size = alphas[i + 1] - alphas[i]
        #     integrated_gradients += step_size * 0.5 * (gradients[i, ...] + gradients[i + 1, ...])
        # return integrated_gradients / gradients.shape[0]

    def execute(self,
                image: torch.Tensor,
                idx_label: int
                ) -> Tensor:
        """Computes the sum of all integrated gradients specified for the input image and label.

        Args:
            image (Tensor): The image for the integrated gradient computation.
            idx_label (int): The label index for the integrated gradient computation.

        Returns:
            Tensor: The integrated gradients.

        See Also:
            - Implementation acc. to `Axiomatic Attribution for Deep Networks <https://arxiv.org/pdf/1703.01365.pdf>`_
        """
        # image.requires_grad = False
        integrated_gradients = torch.zeros_like(image, requires_grad=False)
        baseline = self.baseline_generator.execute(image)

        self.image_interpolator.update_parameters(baseline)
        blended_images, alphas = self.image_interpolator.execute(image)
        gradients = torch.zeros_like(blended_images, requires_grad=False)
        for idx_blending in range(blended_images.shape[0]):
            blended_grads = self._compute_gradients(blended_images[idx_blending, ...],
                                                    0,
                                                    idx_label)
            gradients[idx_blending, ...] = blended_grads
        integral_approximation = self._integral_approximation(gradients, alphas)
        integrated_gradients += (image - baseline) * integral_approximation
        del gradients
        return integrated_gradients


class SoftmaxAlgorithm(AttributionAlgorithm):

    def __init__(self,
                 model: Module
                 ) -> None:
        """Represents an AttributionAlgorithm returning the Softmax of the models output.

        Args:
            model (Module): The model for the softmax computation.
        """
        super().__init__(model)
        self.softmax = torch.nn.Softmax2d()

    def execute(self,
                image: Tensor,
                idx_label: int
                ) -> Tensor:
        """Computes the softmax map for the input specified by the image and the label index.

        Args:
            image (Tensor): The image for the softmax map computation.
            idx_label (int): The label index for the softmax map computation.

        Returns:
            Tensor: The softmax map.
        """

        softmax_map = self.softmax(self.model(image))[:, idx_label, ...]
        return softmax_map


class LogitAlgorithm(AttributionAlgorithm):

    def __init__(self,
                 model: Module
                 ) -> None:
        """Represents an AttributionAlgorithm returning the logits of the models output.

        Args:
            model (Module): The model for the logit computation.
        """
        super().__init__(model)

    def execute(self,
                image: Tensor,
                idx_label: int
                ) -> Tensor:
        """Computes the logit map for the input specified by the image and the label index.

        Args:
            image (Tensor): The image for the logit map computation.
            idx_label (int): The label index for the logit map computation.

        Returns:
            Tensor: The logit map.
        """
        return self.model(image)[:, idx_label, ...]


class IG_class(AttributionAlgorithm):

    def __init__(self,
                 model: Module,
                 baseline_generator: AbstractBaselineGenerator,
                 image_interpolator: AbstractImageInterpolator,
                 verbose: bool = True
                 ) -> None:
        """Integrated Gradient attribution algorithm.

        Args:
            model (Module): The model for the gradient computation.
            spatial_iterator (AbstractSpatialIterator): The interpolator for the evaluation locations.
            baseline_generator (AbstractBaselineGenerator): The generator for the baseline.
            image_interpolator (AbstractImageInterpolator): The interpolator for the generation of a blending.
            verbose (bool): If true the algorithm reports the progress to stdout.
        """
        super().__init__(model)
        self.baseline_generator = baseline_generator
        self.image_interpolator = image_interpolator
        self.verbose = verbose
        # self.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # self.softmax_fn = Softmax2d()

    def _compute_gradients(self,
                           image: Tensor,
                           batch_idx: int,
                           channel_idx: int,
                           ) -> Tensor:
        """Computes the gradients of the specified output w.r.t. the input image.

        Args:
            image (Tensor): The input image to the model.
            batch_idx (int): The batch index specifying the output for gradient computation.
            channel_idx (int): The channel index / label index specifying the output for gradient computation.


        Returns:
            Tensor: The tensor with the gradients of a specified output entry w.r.t. the input image.
        """
        # print(image.shape)
        probs = (self.model(image)).softmax(dim=-1)
        # print(probs.shape)
        grads = torch.autograd.grad(probs[batch_idx, channel_idx], image)[0].detach()
        # print (grads.shape)
                                    # grad_outputs=torch.ones_like(probs[batch_idx, channel_idx, :, :]))[0].detach()
        return grads

    @staticmethod
    def _integral_approximation(gradients: Tensor,
                                alphas: Tensor,
                                ) -> Tensor:
        """Computes the integral approximation based on the Riemann trapezoidal approach.

        Args:
            gradients (Tensor): The gradients of multiple backward steps collected in one tensor.
            alphas (Tensor): The step size between the gradients.

        Returns:
            Tensor: The normalized integral approximation of the tensors.
        """
        integrated_gradients = torch.zeros((gradients.shape[1:]), device=gradients.device)
        for i in range(gradients.shape[0] - 1):
            step_size = alphas[i + 1] - alphas[i]
            integrated_gradients += step_size * 0.5 * (gradients[i, ...] + gradients[i + 1, ...])
        return integrated_gradients / gradients.shape[0]

    def execute(self,
                image: torch.Tensor,
                idx_label: int
                ) -> Tensor:
        """Computes the sum of all integrated gradients specified for the input image and label.

        Args:
            image (Tensor): The image for the integrated gradient computation.
            idx_label (int): The label index for the integrated gradient computation.

        Returns:
            Tensor: The integrated gradients.

        See Also:
            - Implementation acc. to `Axiomatic Attribution for Deep Networks <https://arxiv.org/pdf/1703.01365.pdf>`_
        """
        # image.requires_grad = False
        integrated_gradients = torch.zeros_like(image, requires_grad=False)
        baseline = self.baseline_generator.execute(image)

        self.image_interpolator.update_parameters(baseline)
        blended_images, alphas = self.image_interpolator.execute(image)
        gradients = torch.zeros_like(blended_images, requires_grad=False)
        for idx_blending in range(blended_images.shape[0]):
            blended_grads = self._compute_gradients(blended_images[idx_blending, ...],
                                                    0,
                                                    idx_label)
            gradients[idx_blending, ...] = blended_grads
        integral_approximation = self._integral_approximation(gradients, alphas)
        integrated_gradients += (image - baseline) * integral_approximation
        del gradients
        return integrated_gradients
