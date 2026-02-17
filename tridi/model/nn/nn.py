"""
Implementation of nearest neighbours baseline for Object pop-up with and without class prediction.
"""
from typing import Union, Dict

import numpy as np
try:
    import faiss
except ImportError:
    pass
from scipy.spatial import KDTree

from config.config import ProjectConfig
from tridi.model.nn.common import (
    get_hdf5_files_for_nn, get_sequences_for_nn, get_features_for_nn
)


class KnnWrapper:
    def __init__(
        self,
        model_features, model_labels,
        model_type, backend="faiss_cpu"
    ):
        assert model_features in [
            "human_joints", "human_pose", "human_pose_shape", "human_parameters"
        ]
        assert model_labels in ["data_source", "object_pose", "human_parameters"]
        assert model_type in ["class_specific", "general"]

        self.backend = backend
        self.model_type = model_type
        self.model_features = model_features
        self.model_labels = model_labels

        self.index = None
        self.data = None
        self.labels = None

    def create_index(
        self,
        features: Union[np.ndarray, Dict[int, np.ndarray]],
        labels: Union[np.ndarray, Dict[int, np.ndarray]]
    ):
        if self.model_type == 'general':
            if self.backend == 'scipy':
                self.index = KDTree(features, copy_data=True)
            elif self.backend == 'faiss_cpu':
                self.index = faiss.IndexFlatL2(features.shape[1])
                self.index.add(features.astype(np.float32))
            else:
                resources = faiss.StandardGpuResources()
                flat_config = faiss.GpuIndexFlatConfig()
                flat_config.device = 0

                self.index = faiss.GpuIndexFlatL2(resources, features.shape[1], flat_config)
                self.index.add(features.astype(np.float32))
        else:
            self.index = dict()

            for class_id in features.keys():
                if self.backend == 'scipy':
                    self.index[class_id] = KDTree(features[class_id], copy_data=True)
                elif self.backend == 'faiss_cpu':
                    self.index[class_id] = faiss.IndexFlatL2(features[class_id].shape[1])
                    self.index[class_id].add(features[class_id].astype(np.float32))
                else:
                    resources = faiss.StandardGpuResources()
                    flat_config = faiss.GpuIndexFlatConfig()
                    flat_config.device = 0

                    self.index[class_id] = faiss.GpuIndexFlatL2(resources, features[class_id].shape[1], flat_config)
                    self.index[class_id].add(features[class_id].astype(np.float32))
        self.data = features
        self.labels = labels

    def query(self, features, k=1, class_id=None):
        if self.model_type == 'class_specific':
            assert class_id is not None
        knn = self.index if self.model_type == 'general' else self.index[class_id]

        # ---- NEW: make features always a 2D numpy array ----
        if isinstance(features, list):
            # list of arrays -> concatenate
            feats = []
            for f in features:
                if f is None:
                    continue
                f = np.asarray(f)
                if f.size == 0:
                    continue
                if f.ndim == 1:
                    f = f.reshape(1, -1)
                feats.append(f)
            if len(feats) == 0:
                return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)
            features = np.concatenate(feats, axis=0)
        else:
            features = np.asarray(features)
            if features.ndim == 1:
                features = features.reshape(1, -1)

        if features.shape[0] == 0:
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)
        # -----------------------------------------------

        if self.backend == "scipy":
            distances, indices = knn.query(features.astype(np.float32), k=k)
        else:
            distances, indices = knn.search(features.astype(np.float32), k=k)

        return distances, indices



def create_nn_model(
    cfg: ProjectConfig,
    knn: KnnWrapper,
    train_datasets: list,
    test_datasets: list,
):

    # ===> 1. Locate hdf5 files
    train_hdf5 = get_hdf5_files_for_nn(cfg, train_datasets)
    test_hdf5 = get_hdf5_files_for_nn(cfg, test_datasets)
    # <===

    # ===> 2. Get train / test sequences
    # "dataset": List[(sbj, second_sbj)]
    train_sequences = get_sequences_for_nn(cfg, train_datasets, train_hdf5)
    test_sequences = get_sequences_for_nn(cfg, test_datasets, test_hdf5)

    # Keep a merged view for eval-time feature transforms that may need
    # reference metadata regardless of whether it comes from train or test side.
    knn._all_eval_hdf5_files = {**train_hdf5, **test_hdf5}
    knn._all_eval_sequences = {**train_sequences, **test_sequences}

    # ===> 3. Load features
    if knn.model_type == 'general':
        train_features, train_labels, train_t_stamps = get_features_for_nn(
            knn, train_sequences, train_hdf5, is_train=True
        )
        test_queries, test_labels, test_t_stamps = get_features_for_nn(
            knn, test_sequences, test_hdf5, is_train=False
        )
    else:
        raise NotImplementedError("Class-specific NN not supported anymore.")

    # build kdtree on training data
    knn.create_index(train_features, train_labels)

    return knn, test_queries, test_labels, test_t_stamps


def create_nn_baseline(
    cfg: ProjectConfig,
    model_type: str = 'general',
    sample_target: str = "sbj",  # or "second_sbj"
    backend: str = 'faiss_gpu'
):
    # Initialize wrapper
    knn = KnnWrapper(
        model_features="human_joints",
        model_labels="human_parameters",
        model_type=model_type,
        backend=backend
    )

    # create dataset, split lists
    train_datasets = [(dataset, "train") for dataset in cfg.run.datasets]
    test_datasets = [(dataset, "test") for dataset in cfg.run.datasets]

    knn, test_queries, test_labels, test_t_stamps = create_nn_model(
        cfg, knn, train_datasets, test_datasets
    )

    return knn, test_queries, test_labels, test_t_stamps
