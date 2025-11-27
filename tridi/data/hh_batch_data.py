from dataclasses import dataclass, field, fields
from typing import List, Optional, Union

import torch
from torch import Tensor


@dataclass
class HHBatchData:
    # info
    sbj: Union[str, List[str], None]
    path: Union[str, List[str], None] = None
    t_stamp: Union[int, List[int], None] = None
    # sbj
    sbj_shape: Optional[Tensor] = None  # dim is 10
    sbj_global: Optional[Tensor] = None  # R is 6d
    sbj_pose: Optional[Tensor] = None  # R is 6d => dim is 51*6
    sbj_c: Optional[Tensor] = None  # dim is 3
    sbj_vertices: Optional[Tensor] = None  # dim is 6890x3
    sbj_joints: Optional[Tensor] = None  # dim is 6890x3
    sbj_gender: Optional[Tensor] = None  # dim is 1
    # second sbj
    second_sbj_shape: Optional[Tensor] = None  # dim is 10
    second_sbj_global: Optional[Tensor] = None  # R is 6d
    second_sbj_pose: Optional[Tensor] = None  # R is 6d => dim is 51*6
    second_sbj_c: Optional[Tensor] = None  # dim is 3
    second_sbj_vertices: Optional[Tensor] = None  # dim is 6890x3
    second_sbj_joints: Optional[Tensor] = None  # dim is 6890x3
    second_sbj_gender: Optional[Tensor] = None  # dim is 1
    scale: Optional[Tensor] = None

    meta: dict = field(default_factory=lambda: {})

    def to_string(self):
        info = f"HHBatchData: sbj={self.sbj}, t_stamp={self.t_stamp}, "
        info += f"sbj_shape={None if self.sbj_shape is None else self.sbj_shape.shape}, "
        info += f"sbj_global={None if self.sbj_global is None else self.sbj_global.shape}, "
        info += f"sbj_pose={None if self.sbj_pose is None else self.sbj_pose.shape}, "
        info += f"sbj_c={None if self.sbj_c is None else self.sbj_c.shape}, "
        info += f"sbj_vertices={None if self.sbj_vertices is None else self.sbj_vertices.shape}, "
        info += f"sbj_joints={None if self.sbj_joints is None else self.sbj_joints.shape}, "
        info += f"second_sbj_shape={None if self.second_sbj_shape is None else self.second_sbj_shape.shape}, "
        info += f"second_sbj_global={None if self.second_sbj_global is None else self.second_sbj_global.shape}, "
        info += f"second_sbj_pose={None if self.second_sbj_pose is None else self.second_sbj_pose.shape}, "
        info += f"second_sbj_c={None if self.second_sbj_c is None else self.second_sbj_c.shape}, "
        info += f"second_sbj_vertices={None if self.second_sbj_vertices is None else self.second_sbj_vertices.shape}, "
        info += f"second_sbj_joints={None if self.second_sbj_joints is None else self.second_sbj_joints.shape}, "
        return info

    def to(self, *args, **kwargs):
        new_params = {}
        for field_name in iter(self):
            value = getattr(self, field_name)
            if isinstance(value, (torch.Tensor)):
                new_params[field_name] = value.to(*args, **kwargs)
            else:
                new_params[field_name] = value
        batch_data = type(self)(**new_params)
        return batch_data

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    def batch_size(self):
        for f in iter(self):
            if f != "meta":
                attr = self.__getattribute__(f)
                if not (attr is None):
                    return len(attr)

    # the following functions make sure **batch_data can be passed to functions
    def __iter__(self):
        for f in fields(self):
            if f.name.startswith("_"):
                continue

            yield f.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return sum(1 for f in iter(self))

    @classmethod
    def collate(cls, batch):
        """
        Given a list objects `batch` of class `cls`, collates them into a batched
        representation suitable for processing with deep networks.
        """

        elem = batch[0]

        if isinstance(elem, cls):
            collated = {}
            for f in fields(elem):
                if not f.init:
                    continue

                list_values = [getattr(d, f.name) for d in batch]
                
                collated[f.name] = (
                    cls.collate(list_values)
                    if all(list_value is not None for list_value in list_values)
                    else None
                )
            return cls(**collated)
        else:
            return torch.utils.data._utils.collate.default_collate(batch)
