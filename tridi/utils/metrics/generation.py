"""
S_r - a test set
S_g - generated set, |S_g|=|S_r|
"""
from pathlib import Path
from typing import Union

import numpy as np

from tridi.model.nn.nn import KnnWrapper, create_nn_model
from config.config import ProjectConfig


def sample_target_to_nn_feature(sample_target: str):
    """
    Map evaluator's sampling_target to NN feature type.
    Your evaluator passes: "sbj" / "second_sbj" (and maybe "obj" later).
    """
    # multi-human: both are human joints features
    if sample_target in ["sbj", "second_sbj", "human"]:
        return "human_joints"

    # optional: object branch (if you ever add it back)
    if sample_target in ["obj", "object"]:
        return "object_pose"

    # optional combined mode
    if "sbj_obj" in sample_target:
        return "human_joints_object_pose"

    raise ValueError(f"Unknown sample_target: {sample_target}")


def _to_2d_float32(x):
    """
    Make sure queries passed into knn.query are np.ndarray float32.
    Fixes: AttributeError: 'list' object has no attribute 'astype'
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)

    if isinstance(x, list):
        if len(x) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        # common case: list of arrays -> concat
        if isinstance(x[0], np.ndarray):
            return np.concatenate(x, axis=0).astype(np.float32)
        # list of scalars
        return np.asarray(x, dtype=np.float32)

    return np.asarray(x, dtype=np.float32)


def coverage(
    cfg: ProjectConfig,
    samples_file: Union[str, Path],
    reference_dataset: str,
    reference_set="test",          # or "train"
    sample_target="sbj",           # "sbj" or "second_sbj"
):
    # NN:
    #   query: S_g
    #   train set: S_r
    cfg.sample.samples_file = samples_file

    knn = KnnWrapper(
        model_features=sample_target_to_nn_feature(sample_target),
        model_labels="data_source",
        model_type="general",
        backend="faiss_cpu"
    )
    # <<< IMPORTANT: tell common.py which branch to read
    knn.sample_target = sample_target

    train_datasets = [(reference_dataset, reference_set)]
    test_datasets = [("samples", "test")]

    knn, test_queries, test_labels, test_t_stamps = create_nn_model(
        cfg, knn, train_datasets, test_datasets
    )

    queries = _to_2d_float32(test_queries["samples"])
    _, pred_indices = knn.query(queries, k=1)

    unique_indices = np.unique(pred_indices)
    cov = len(unique_indices) / len(knn.labels)
    return cov


def minimum_matching_distance(
    cfg: ProjectConfig,
    samples_file: Union[str, Path],
    reference_dataset: str,
    reference_set="test",          # or "train"
    sample_target="sbj",           # "sbj" or "second_sbj"
):
    # NN:
    #   query: S_r
    #   train set: S_g
    cfg.sample.samples_file = samples_file

    knn = KnnWrapper(
        model_features=sample_target_to_nn_feature(sample_target),
        model_labels="data_source",
        model_type="general",
        backend="faiss_cpu"
    )
    knn.sample_target = sample_target

    train_datasets = [("samples", "test")]
    test_datasets = [(reference_dataset, reference_set)]

    knn, test_queries, test_labels, test_t_stamps = create_nn_model(
        cfg, knn, train_datasets, test_datasets
    )

    queries = _to_2d_float32(test_queries[reference_dataset])
    pred_distances, _ = knn.query(queries, k=1)

    return float(np.sum(pred_distances) / len(queries))


def nearest_neighbor_accuracy(
    cfg: ProjectConfig,
    samples_file: Union[str, Path],
    reference_dataset: str,
    compare_against="test",        # or "train"
    sample_target="sbj",           # "sbj" or "second_sbj"
    subsample: bool = False,
):
    """
    1-NNA between reference set and generated set.
    """
    datasets = [reference_dataset, "samples"]
    cfg.sample.samples_file = samples_file

    knn = KnnWrapper(
        model_features=sample_target_to_nn_feature(sample_target),
        model_labels="data_source",
        model_type="general",
        backend="faiss_cpu"
    )
    knn.sample_target = sample_target

    train_datasets = [(d, compare_against) for d in datasets]
    test_datasets = [(d, compare_against) for d in datasets]

    knn, test_queries, test_labels, test_t_stamps = create_nn_model(
        cfg, knn, train_datasets, test_datasets
    )

    generated_total, reference_total = 0, 0
    generated_hits, reference_hits = 0, 0

    for d in datasets:
        queries = _to_2d_float32(test_queries[d])
        labels = np.asarray(test_labels[d])

        # k=2: [0] is itself, [1] is nearest other
        _, pred_indices = knn.query(queries, k=2)
        pred_labels = knn.labels[pred_indices[:, 1]]

        if d == "samples":
            generated_total += len(queries)
            generated_hits += int(np.sum(pred_labels == labels))
        else:
            reference_total += len(queries)
            reference_hits += int(np.sum(pred_labels == labels))

    nna = (generated_hits + reference_hits) / (generated_total + reference_total)
    return float(nna)


def sample_distance(
    cfg: ProjectConfig,
    samples_file: Union[str, Path],
    reference_dataset: str,
    reference_set="train",
    sample_target="sbj",           # "sbj" or "second_sbj"
):
    # NN:
    #   query: S_g
    #   train set: S_train
    cfg.sample.samples_file = samples_file

    knn = KnnWrapper(
        model_features=sample_target_to_nn_feature(sample_target),
        model_labels="data_source",
        model_type="general",
        backend="faiss_cpu"
    )
    knn.sample_target = sample_target

    train_datasets = [(reference_dataset, reference_set)]
    test_datasets = [("samples", "test")]

    knn, test_queries, test_labels, test_t_stamps = create_nn_model(
        cfg, knn, train_datasets, test_datasets
    )

    queries = _to_2d_float32(test_queries["samples"])
    pred_distances, _ = knn.query(queries, k=1)

    return float(np.sum(pred_distances) / len(queries))
