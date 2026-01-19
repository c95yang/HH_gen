from pathlib import Path

import torch

from tridi.data.hh_dataset import HHDataset
from tridi.data.hh_batch_data import HHBatchData
from config.config import ProjectConfig

# logger = get_logger(__name__)
from logging import getLogger
logger = getLogger(__name__)


def get_train_dataloader(cfg: ProjectConfig):
    train_datasets, val_datasets = [], []

    for dataset_name in cfg.run.datasets:
        if dataset_name == 'behave':
            dataset_config = cfg.behave
            train_kwargs = {"behave_repeat_fix": True, "split_file": cfg.behave.train_split_file}
            val_split = "test"
            val_kwargs = {"split_file": cfg.behave.test_split_file}

        elif dataset_name == "embody3d":
            dataset_config = cfg.embody3d
            train_kwargs = {"split_file": cfg.embody3d.train_split_file}
            val_split = "test"
            val_kwargs = {"split_file": cfg.embody3d.test_split_file}

        elif dataset_name == "interhuman":
            dataset_config = cfg.interhuman
            train_kwargs = {"split_file": cfg.interhuman.train_split_file}
            val_split = "test"
            val_kwargs = {"split_file": cfg.interhuman.test_split_file}

        elif dataset_name == "chi3d":
            dataset_config = cfg.chi3d
            train_kwargs = {"split_file": cfg.chi3d.train_split_file}
            val_split = "val"
            val_kwargs = {"split_file": cfg.chi3d.val_split_file}

        else:
            raise NotImplementedError(f'Unknown dataset: {dataset_name}')

        train_dataset = HHDataset(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            split='train',
            downsample_factor=dataset_config.downsample_factor,
            subjects=getattr(dataset_config, "train_subjects", []),
            augment_rotation=dataset_config.augment_rotation if cfg.run.job == "train" else False,
            augment_symmetry=dataset_config.augment_symmetry if cfg.run.job == "train" else False,
            assets_folder=Path(cfg.env.assets_folder),
            fps=dataset_config.fps_train,
            max_timestamps=getattr(dataset_config, "max_timestamps", None),
            filter_subjects=getattr(dataset_config, "filter_subjects", None),
            **train_kwargs
        )     

        val_dataset = HHDataset(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            split=val_split,  # chi3d=val，其它=test
            downsample_factor=dataset_config.downsample_factor,
            subjects=getattr(dataset_config, f"{val_split}_subjects", None),
            assets_folder=Path(cfg.env.assets_folder),
            fps=dataset_config.fps_eval,
            max_timestamps=getattr(dataset_config, "max_timestamps", None),
            filter_subjects=getattr(dataset_config, "filter_subjects", None),
            **val_kwargs
        )

        # accumulate datasets
        # canonical_obj_meshes.update(train_dataset.canonical_obj_meshes)
        # canonical_obj_keypoints.update(train_dataset.canonical_obj_keypoints)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    # concatenate datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    if cfg.dataloader.sampler == "weighted":
        train_dataset_length = len(train_dataset)
        train_weights = []
        for dataset in train_datasets:
            dataset_length = len(dataset)
            weights = torch.ones(dataset_length, dtype=torch.double)
            weights = (dataset_length / train_dataset_length) * weights
            train_weights.append(weights)
        train_weights = torch.cat(train_weights, dim=0)

        sampler = torch.utils.data.WeightedRandomSampler(
            train_weights,
            num_samples=min(100000, train_dataset_length),
            replacement=False
        )
    elif cfg.dataloader.sampler == "random":
        sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=50000)
    elif cfg.dataloader.sampler == "default":
        sampler = None
    else:
        raise NotImplementedError(f"Unknown sampler: {cfg.dataloader.sampler}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.workers,
        drop_last=False,
        sampler=sampler,
        pin_memory=True,
        collate_fn=HHBatchData.collate,
        persistent_workers=False,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=HHBatchData.collate,
        persistent_workers=False,
    )
    logger.info(f"Train data length: {len(train_dataset)}")
    logger.info(f"Val data length: {len(val_dataset)}")

    # return train_dataloader, val_dataloader, canonical_obj_meshes, canonical_obj_keypoints
    return train_dataloader, val_dataloader


def get_eval_dataloader(cfg: ProjectConfig):
    datasets = []

    if cfg.run.job == "sample":
        split = getattr(cfg.sample, "split", "test")
    else:  # run.job == "eval"
        split = getattr(cfg.eval, "split", "test")

    for dataset_name in cfg.run.datasets:
        if dataset_name == "chi3d":
            dataset_config = cfg.chi3d
            # auto choose chi3d_train/val/test.json
            split_file = getattr(dataset_config, f"{split}_split_file")

            dataset = HHDataset(
                name=dataset_config.name,
                root=Path(dataset_config.root),
                split=split,                       # <-- open dataset_{split}_{fps}fps.hdf5
                augment_symmetry=dataset_config.augment_symmetry,
                augment_rotation=dataset_config.augment_rotation,
                downsample_factor=1,
                subjects=getattr(dataset_config, f"{split}_subjects", None),
                assets_folder=Path(cfg.env.assets_folder),
                fps=dataset_config.fps_eval,
                max_timestamps=dataset_config.max_timestamps,
                filter_subjects=dataset_config.filter_subjects,
                split_file=split_file,
            )
        else:
            # only chi3d
            raise NotImplementedError(f"Only chi3d is supported in your current setup, got: {dataset_name}")

        datasets.append(dataset)

    dataloaders = []
    for dataset in datasets:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=HHBatchData.collate,
            persistent_workers=False,
        )
        dataloaders.append(dataloader)
        logger.info(f"Eval data length for {dataset.name} ({split}): {len(dataset)}")

    return dataloaders