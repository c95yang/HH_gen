from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os

from .environment import EnvironmentConfig


# ========================== Preprocessing configs =============================
@dataclass
class PreprocessBehaveConfig:
    # paths
    root: str = os.path.join("${env.raw_datasets_folder}", "behave/")
    target: str = os.path.join("${env.datasets_folder}", "behave_smplh/")
    orig_objects_path: str = os.path.join("${behave.root}", "objects")
    can_objects_path: str = os.path.join("${behave.root}", "canonicalized_objects")
    full_anno_path: str = os.path.join("${behave.root}", "behave-30fps-params-v1")

    # Split (needed to correctly set resulting filename and to load sequences from split file)
    split: str = "train"  # "train", "test"
    # Path to a split_file (for selecting sequences based on split)
    split_file: str = os.path.join("${behave.root}", "split.json")
    # Useful to preprocess sequences for basketball and keyboard that only have 1fps annotations
    # together with 30fps sequences
    combine_1fps_and_30fps: bool = True
    # Downsample from to 10fps or 1fps, only applicable to 30fps data
    downsample: str = "10fps"
    # Preprocess only selected subjects
    subjects: List = field(default_factory=lambda: ["*"])
    # Preprocess only selected objects
    objects: List = field(default_factory=lambda: ["yogamat"])
    # Threshold for computing human-object contacts (if no precomputed contacts are provided)
    contact_threshold: float = 0.020
    # Generate object keypoints after preprocessing
    generate_obj_keypoints: bool = False

@dataclass
class PreprocessEmbody3DConfig:
    # paths
    root: str = os.path.join("${env.raw_datasets_folder}", "embody3d/")
    target: str = os.path.join("${env.datasets_folder}", "embody3d_smplx/")
    assets:str = "${env.assets_folder}"
    # Split (needed to correctly set resulting filename)
    split: str = "train"  # "train", "test"
    # Path to a split_file (for selecting sequences based on split)
    split_file: str = os.path.join("${embody3d.root}", "split.json")
    downsample: str = "10fps" # "None", "30fps", "10fps", "1fps"
    # Preprocess only selected categaories
    categories: List = field(default_factory=lambda: ["*"])

@dataclass
class PreprocessInterHumanConfig:
    # paths
    root: str = os.path.join("${env.raw_datasets_folder}", "interhuman/")
    target: str = os.path.join("${env.datasets_folder}", "interhuman_smpl/")

    # Split (needed to correctly set resulting filename)
    split: str = "train"  # "train", "test"
    # Path to a split_file (for selecting sequences based on split)
    split_file: str = os.path.join("${interhuman.root}", "split.json")
    downsample: str = "10fps" # "None", "30fps", "10fps", "1fps"

@dataclass
class PreprocessCHI3DConfig:
    # paths
    root: str = os.path.join("${env.raw_datasets_folder}", "chi3d/")
    target: str = os.path.join("${env.datasets_folder}", "chi3d_smplx/")
    assets:str = "${env.assets_folder}"
    # Split (needed to correctly set resulting filename)
    split: str = "train"  # "train", "test"
    # Path to a split_file (for selecting sequences based on split)
    split_file: str = os.path.join("${chi3d.root}", "split.json")
    downsample: str = "10fps" # "None", "50fps", "10fps", "1fps"
    # Split strategy used to generate split.json and asset lists.
    # Supported: "scenario4" (legacy), "split5" (interaction-aware, reduced weak edges).
    split_strategy: str = "scenario4"
    # split5: per <subject, motion> validation ratio sampled from non-test sequences.
    split5_val_ratio: float = 0.2
    # split5: interactions with fewer edges than this are considered weak.
    split5_min_contact_edges: int = 2
    # split5: enable frame-level active-window filtering using interaction signatures.
    split5_use_active_window: bool = True
    # split5: margin around active window, expressed as a ratio of active length.
    split5_window_margin_ratio: float = 0.2
    # split5: minimum kept frames after windowing (before downsampling).
    split5_window_min_frames: int = 64
    # split5: if no frame-level signal is found for a sequence, keep full sequence.
    split5_keep_full_if_no_signal: bool = True

@dataclass
class PreprocessGrabConfig:
    # paths
    root: str = os.path.join("${env.raw_datasets_folder}", "grab/")
    target: str = os.path.join("${env.datasets_folder}", "grab_smplh/")
    objects_path: str = os.path.join("${grab.root}", "tools/object_meshes/decimated_meshes")

    # Split (needed to correctly set resulting filename)
    split: str = "train"  # "train", "test"
    # Preprocess only frames with human-object contacts
    only_contact_frames: bool = True
    # Downsample from 120fps to 30fps or 10fps
    downsample: str = "10fps" # "None", "30fps", "10fps", "1fps"
    # Use modified meshes with fewer vertices
    use_decimated_obj_meshes: bool = True
    # Preprocess only selected subjects
    subjects: List = field(default_factory=lambda: ["*"])
    # Preprocess only selected objects
    objects: List = field(default_factory=lambda: [
        "banana", "binoculars", "camera", "coffeemug",
        "cup", "doorknob", "eyeglasses", "flute",
        "flashlight", "fryingpan", "gamecontroller", "hammer",
        "headphones", "knife", "lightbulb", "mouse",
        "mug", "phone", 'teapot', "toothbrush", "wineglass"
    ])
    # Load generated SMPL-h meshes (works only with SMPL-h)
    load_existing_sbj_meshes: bool = False
    # Generate object keypoints after preprocessing
    generate_obj_keypoints: bool = False


@dataclass
class PreprocessIntercapConfig:
    # paths
    root: str = os.path.join("${env.raw_datasets_folder}", "intercap/")
    target: str = os.path.join("${env.datasets_folder}", "intercap_smplh/")
    objects_path: str = os.path.join("${intercap.root}", "tools/canonicalized_objects")

    # Split (needed to use pre-defined splits)
    split: str = "train"  # "train", "test"
    # Use manualy canonicalized meshes (False is not supported)
    use_canonicalized_obj_meshes: bool = True
    # Downsample to 1fps
    downsample: bool = True
    # Preprocess only selected subjects
    subjects: List = field(default_factory=lambda: [])
    # Preprocess only selected objects
    objects: List = field(default_factory=lambda: [])
    # Threshold for computing human-object contacts (if no precomputed contacts are provided)
    contact_threshold: float = 0.050
    # Custom object names
    object_names: Dict[str, str] = field(default_factory=lambda: {
        "obj_01": "suitcase", "obj_02": "skateboard", "obj_03": "ball",
        "obj_04": "umbrella", "obj_05": "tennisracket", "obj_06": "briefcase",
        "obj_07": "chair", "obj_08": "bottle", "obj_09": "cup", "obj_10": "ottoman"
    })
    # Generate object keypoints after preprocessing
    generate_obj_keypoints: bool = False


@dataclass
class PreprocessOmomoConfig:
    # paths
    root: str = os.path.join("${env.raw_datasets_folder}", "omomo/")
    target: str = os.path.join("${env.datasets_folder}", "omomo_smplh/")
    objects_path: str = os.path.join("${omomo.root}", "tools/canonicalized_decimated_objects")

    # Split (needed to use pre-defined splits)
    split: str = "train"  # "train", "test"
    # Use manualy canonicalized meshes (False is not supported)
    use_canonicalized_obj_meshes: bool = True
    # Downsample to 1fps
    downsample: bool = True
    # Preprocess only selected subjects
    subjects: List = field(default_factory=lambda: [])
    # Preprocess only selected objects
    objects: List = field(default_factory=lambda: [])
    # Threshold for computing human-object contacts (if no precomputed contacts are provided)
    contact_threshold: float = 0.050
    # Generate object keypoints after preprocessing
    generate_obj_keypoints: bool = False
# ==============================================================================

@dataclass
class PreprocessConfig:
    # environment
    env: EnvironmentConfig = EnvironmentConfig()

    # Type of data to process
    input_type: str = "smplh"  # smplh, hands, smpl
    # Number of points to sample from object mesh
    obj_keypoints_npoints: int = 1500
    # Align SMPL using skeleton information
    align_with_joints: bool = False
    # Scale to [-1, 1]
    normalize: bool = False
    # Align with ground plane
    align_with_ground: bool = False

    # datasets
    behave: PreprocessBehaveConfig = PreprocessBehaveConfig()
    embody3d: PreprocessEmbody3DConfig = PreprocessEmbody3DConfig()
    interhuman: PreprocessInterHumanConfig = PreprocessInterHumanConfig()
    chi3d: PreprocessCHI3DConfig = PreprocessCHI3DConfig()
    grab: PreprocessGrabConfig = PreprocessGrabConfig()
    intercap: PreprocessIntercapConfig = PreprocessIntercapConfig()
    omomo: PreprocessOmomoConfig = PreprocessOmomoConfig()

    overfit: bool = False
    debug: bool = False
