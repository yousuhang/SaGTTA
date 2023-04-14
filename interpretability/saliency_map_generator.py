from abc import (ABC, abstractmethod)
from typing import (Any, Optional, Tuple)
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from NetworkZoo.Definitions import KEY_IMAGE
from pymia.data.extraction import ParameterizableDataset
from pymia.data.indexexpression import IndexExpression

from .AttributionAlgorithms import (IntegratedGradientAlgorithm, SoftmaxAlgorithm, LogitAlgorithm, AttributionAlgorithm)
from interpretability.utilities import (AbstractBaselineGenerator, SelectiveMinimaBaseline, AbstractImageInterpolator)
from interpretability.selection import (AbstractSpatialIterator)
from interpretability.data import (AbstractModelLoader, BasicHDF5DataSourceLoader, AbstractAttributionSaver)


class AbstractSaliencyMapGenerator(ABC):

    def __init__(self,
                 model_loader: AbstractModelLoader,
                 data_source_loader: BasicHDF5DataSourceLoader,
                 spatial_interpolator: Optional[AbstractSpatialIterator],
                 attribution_saver: Optional[AbstractAttributionSaver],
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.model_loader = model_loader
        self.data_source_loader = data_source_loader
        self.spatial_interpolator = spatial_interpolator
        self.attribution_saver = attribution_saver
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    @abstractmethod
    def test_case(self, *args, **kwargs) -> Any:
        pass


class ParameterizableDatasetSaliencyMapGenerator(AbstractSaliencyMapGenerator):

    def __init__(self,
                 model_loader: AbstractModelLoader,
                 data_source_loader: BasicHDF5DataSourceLoader,
                 spatial_interpolator: Optional[AbstractSpatialIterator],
                 attribution_saver: Optional[AbstractAttributionSaver],
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(model_loader, data_source_loader, spatial_interpolator, attribution_saver, *args, **kwargs)
        self.algorithm_name = 'NoAlgorithm'

    @staticmethod
    def _get_subject_index_by_name(dataset: ParameterizableDataset, subject_name: str):
        dataset_subjects = dataset.get_subjects()

        if subject_name not in dataset_subjects:
            raise ValueError(f'The subject {subject_name} is not contained in the dataset!')

        return dataset_subjects.index(subject_name)

    @staticmethod
    def _expand_tensor_dimensions(x: Tensor,
                                  num_dims: int
                                  ) -> Tensor:
        assert num_dims > x.ndim, f'The given tensor has more dimensions ({x.ndim}) as requested ({num_dims})!'

        while x.ndim < num_dims:
            x = x.unsqueeze(0)

        return x

    def prepare_model_testing(self,
                              subject_name: str,
                              axial_index: int
                              ) -> Tuple[Module, Tensor]:
        model = self.model_loader.load(self.device)

        dataset = self.data_source_loader.load()

        subject_idx = self._get_subject_index_by_name(dataset, subject_name)
        index_expression = IndexExpression(axial_index, axis=0)
        batch = dataset.direct_extract(extractor=self.data_source_loader.extractor,
                                       subject_index=subject_idx,
                                       index_expr=index_expression,
                                       transform=self.data_source_loader.transform)

        image = torch.as_tensor(np.stack(batch[KEY_IMAGE], axis=0), dtype=torch.float, device=self.device)
        image = self._expand_tensor_dimensions(image, num_dims=4)

        return model, image

    def execute_testing(self,
                        algorithm: AttributionAlgorithm,
                        attribution_placeholder: Tensor,
                        image: Tensor,
                        target_indices: Tuple[int, ...],
                        subject_name: str,
                        axial_index: int,
                        algorithm_name: str
                        ) -> Tensor:

        for target_idx in target_indices:
            label_attribution = algorithm.execute(image=image, idx_label=target_idx)

            if self.attribution_saver is not None:
                self.attribution_saver.save(label_attribution, subject_name=subject_name, slice_idx=axial_index,
                                            label_idx=target_idx, prefix=algorithm_name, postfix='slice')

            attribution_placeholder += label_attribution

        if self.attribution_saver is not None:
            self.attribution_saver.save(attribution_placeholder, subject_name=subject_name, slice_idx=axial_index,
                                        label_idx='', prefix=algorithm_name, postfix='all_labels')

        return attribution_placeholder

    @abstractmethod
    def test_case(self,
                  *args,
                  **kwargs
                  ) -> Any:
        pass


class IntegratedGradientsMapGenerator(ParameterizableDatasetSaliencyMapGenerator):

    def __init__(self,
                 model_loader: AbstractModelLoader,
                 data_source_loader: BasicHDF5DataSourceLoader,
                 spatial_interpolator: AbstractSpatialIterator,
                 image_interpolator: AbstractImageInterpolator,
                 attribution_saver: Optional[AbstractAttributionSaver] = None,
                 baseline_interpolator: AbstractBaselineGenerator = SelectiveMinimaBaseline(),
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(model_loader, data_source_loader, spatial_interpolator, attribution_saver, *args, **kwargs)
        self.image_interpolator = image_interpolator
        self.baseline_interpolator = baseline_interpolator
        self.algorithm_name = 'IntegratedGradients'

    def test_case(self,
                  subject_name: str,
                  axial_index: int,
                  target_indices: tuple,
                  *args,
                  **kwargs
                  ) -> Tensor:

        model, image = self.prepare_model_testing(subject_name, axial_index)

        algorithm = IntegratedGradientAlgorithm(model=model,
                                                spatial_iterator=self.spatial_interpolator,
                                                baseline_generator=self.baseline_interpolator,
                                                image_interpolator=self.image_interpolator)

        attribution = torch.zeros_like(image, requires_grad=False)

        attribution = self.execute_testing(algorithm, attribution, image, target_indices, subject_name, axial_index,
                                           self.algorithm_name)

        return attribution


class SoftmaxMapGenerator(ParameterizableDatasetSaliencyMapGenerator):

    def __init__(self,
                 model_loader: AbstractModelLoader,
                 data_source_loader: BasicHDF5DataSourceLoader,
                 attribution_saver: Optional[AbstractAttributionSaver],
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(model_loader, data_source_loader, None, attribution_saver, *args, **kwargs)
        self.algorithm_name = 'SoftmaxMap'

    def test_case(self,
                  subject_name: str,
                  axial_index: int,
                  target_indices: tuple,
                  *args,
                  **kwargs
                  ) -> Tensor:

        model, image = self.prepare_model_testing(subject_name, axial_index)

        algorithm = SoftmaxAlgorithm(model=model)

        zeros_shape = list(image.shape)
        zeros_shape.pop(1)
        attribution = torch.zeros(zeros_shape, requires_grad=False, dtype=image.dtype, device=self.device)

        attribution = self.execute_testing(algorithm, attribution, image, target_indices, subject_name, axial_index,
                                           self.algorithm_name)

        return attribution


class LogitMapGenerator(ParameterizableDatasetSaliencyMapGenerator):

    def __init__(self,
                 model_loader: AbstractModelLoader,
                 data_source_loader: BasicHDF5DataSourceLoader,
                 attribution_saver: Optional[AbstractAttributionSaver],
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(model_loader, data_source_loader, None, attribution_saver, *args, **kwargs)
        self.algorithm_name = 'LogitMap'

    def test_case(self,
                  subject_name: str,
                  axial_index: int,
                  target_indices: tuple,
                  *args,
                  **kwargs
                  ) -> Tensor:

        model, image = self.prepare_model_testing(subject_name, axial_index)

        algorithm = LogitAlgorithm(model)

        zeros_shape = list(image.shape)
        zeros_shape.pop(1)
        attribution = torch.zeros(zeros_shape, requires_grad=False, dtype=image.dtype, device=self.device)

        attribution = self.execute_testing(algorithm, attribution, image, target_indices, subject_name, axial_index,
                                           self.algorithm_name)

        return attribution

