from typing import Dict, Union, List, Optional, Tuple

import numpy as np
import smplx
import torch
import trimesh

from tridi.data.hh_batch_data import HHBatchData
from tridi.model.base import TriDiModelOutput
from tridi.utils.geometry import rotation_6d_to_matrix


class MeshModel:
    """
    A class that generates human and object mesh given the parameters
    predicted by the denoising model.
    """
    def __init__(
        self, model_path: str, batch_size: int,
        # canonical_obj_meshes: Dict[int, trimesh.Trimesh],
        # canonical_obj_keypoints: Dict[int, Dict[str, np.ndarray]],
        device='cpu'
    ) -> None:
        self.smpl_f = smplx.build_layer(
            model_path=model_path, model_type="smplh", gender="female",
            use_pca=False, num_betas=10, batch_size=batch_size
        ).to(device)

        self.smpl_m = smplx.build_layer(
            model_path=model_path, model_type="smplh", gender="male",
            use_pca=False, num_betas=10, batch_size=batch_size
        ).to(device)

        # object meshes and keypoints
        # self.canonical_obj_meshes = canonical_obj_meshes
        # self.canonical_obj_keypoints = canonical_obj_keypoints
        # class_ids = sorted(self.canonical_obj_keypoints.keys())
        # self.canonical_obj_keypoints_th = []
        # self.canonical_obj_normals_th = []
        # for class_id in class_ids:
        #     self.canonical_obj_keypoints_th.append(
        #         self.canonical_obj_keypoints[class_id]["cartesian"]
        #     )
        #     # self.canonical_obj_normals_th.append(
        #     #     self.canonical_obj_keypoints[class_id]["normals"]
        #     # )
        # self.canonical_obj_keypoints_th = torch.tensor(
        #     np.stack(self.canonical_obj_keypoints_th, axis=0), dtype=torch.float,
        #     requires_grad=False, device=device
        # )
        # # self.canonical_obj_normals_th = torch.tensor(
        # #     np.stack(self.canonical_obj_normals_th, axis=0), dtype=torch.float,
        # #     requires_grad=False, device=device
        # # )
        # canonical_obj_pcs: List[torch.Tensor] = []
        # for class_id in class_ids:
        #     canonical_obj_pcs.append(
        #         torch.tensor(
        #             self.canonical_obj_meshes[class_id].vertices,
        #             dtype=torch.float, requires_grad=False, device=device
        #         )
        #     )
        # self.canonical_obj_pcs = canonical_obj_pcs

        # batch size
        self.batch_size = batch_size
        self.device = device

    @staticmethod
    def _apply_scale(
            sbj_vertices: torch.Tensor,
            sbj_joints: torch.Tensor,
            scale: Optional[Union[torch.Tensor, float]] = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if scale is not None:
            if not isinstance(scale, torch.Tensor):
                scale = torch.tensor(scale, dtype=torch.float, requires_grad=False)
            else:
                scale = scale.clone().detach().float()

            scale = scale.to(sbj_vertices.device).reshape(-1, 1, 1)
            sbj_vertices *= scale
            sbj_joints *= scale

        return sbj_vertices, sbj_joints

    def get_meshes_th(
        self, output: TriDiModelOutput, obj_class, scale=1.0,
        sbj_gender=None, return_joints=False
    ):
        B = min(self.batch_size, len(output))

        sbj_pose = output.sbj_pose.reshape(B, -1, 6)
        sbj_pose = rotation_6d_to_matrix(sbj_pose).reshape(B, -1, 9)
        sbj_global = output.sbj_global.reshape(B, 1, 6)
        sbj_global = rotation_6d_to_matrix(sbj_global).reshape(B, 1, 9)
        body_model_params = {
            "betas": output.sbj_shape,
            "transl": output.sbj_c,
            "global_orient": sbj_global,
            "body_pose": sbj_pose[:, :21],
            "left_hand_pose": sbj_pose[:, 21: 21 + 15],
            "right_hand_pose": sbj_pose[:, 21 + 15:]
        }

        # get sbj mesh
        sbj_vertices, sbj_joints = self.get_smpl_th(body_model_params, sbj_gender)

        # obj mesh
        obj_R = output.obj_R.reshape(B, 1, 6)
        obj_R = rotation_6d_to_matrix(obj_R).reshape(B, 3, 3)

        # we use list here to get rid of dependency on Pointclouds from pytorch3d
        obj_cls_list = obj_class.cpu().tolist()
        obj_verts_list = [self.canonical_obj_pcs[int(c)].clone() for c in obj_cls_list]  # each is (n_i, 3)
        max_n = max(v.shape[0] for v in obj_verts_list)
        obj_vertices = torch.zeros((B, max_n, 3), dtype=obj_verts_list[0].dtype)
        for i, v in enumerate(obj_verts_list):
            obj_vertices[i, : v.shape[0]] = v
        obj_vertices = obj_vertices.to(self.device)

        # Pose object
        obj_vertices = torch.bmm(obj_R, obj_vertices.transpose(2, 1)).transpose(2, 1)
        obj_vertices += output.obj_c.unsqueeze(1)

        # Optionally apply scale
        sbj_vertices, obj_vertices, sbj_joints = self._apply_scale(
            sbj_vertices, obj_vertices, sbj_joints, scale
        )

        if return_joints:
            return sbj_vertices, obj_vertices, sbj_joints
        else:
            return sbj_vertices, obj_vertices

    def get_meshes_wkpts_th(
        self, output: TriDiModelOutput, scale=1.0,
        sbj_gender=None, return_joints=False
    ):
        B = min(self.batch_size, len(output))

        # print("Output sbj_pose shape:", output.sbj_pose.shape)
        # print("Output second_sbj_pose shape:", output.second_sbj_pose.shape)

        sbj_pose = output.sbj_pose.reshape(B, -1, 6)
        sbj_pose = rotation_6d_to_matrix(sbj_pose).reshape(B, -1, 9)
        sbj_global = output.sbj_global.reshape(B, 1, 6)
        sbj_global = rotation_6d_to_matrix(sbj_global).reshape(B, 1, 9)
        body_model_params = {
            "betas": output.sbj_shape,
            "transl": output.sbj_c,
            "global_orient": sbj_global,
            "body_pose": sbj_pose[:, :21],
            "left_hand_pose": sbj_pose[:, 21: 21 + 15],
            "right_hand_pose": sbj_pose[:, 21 + 15:]
        }
        sbj_vertices, sbj_joints = self.get_smpl_th_single(body_model_params, sbj_gender)

        second_sbj_pose = output.second_sbj_pose.reshape(B, -1, 6)
        second_sbj_pose = rotation_6d_to_matrix(second_sbj_pose).reshape(B, -1, 9)
        second_sbj_global = output.second_sbj_global.reshape(B, 1, 6)
        second_sbj_global = rotation_6d_to_matrix(second_sbj_global).reshape(B, 1, 9)
        second_body_model_params = {
            "betas": output.second_sbj_shape,
            "transl": output.second_sbj_c,
            "global_orient": second_sbj_global,
            "body_pose": second_sbj_pose[:, :21],
            "left_hand_pose": second_sbj_pose[:, 21: 21 + 15],
            "right_hand_pose": second_sbj_pose[:, 21 + 15:]
        }
        # get second sbj mesh
        second_sbj_vertices, second_sbj_joints = self.get_smpl_th_single(second_body_model_params, sbj_gender)

        if return_joints:
            return sbj_vertices, sbj_joints, second_sbj_vertices, second_sbj_joints
        else:
            return sbj_vertices, second_sbj_vertices

    def get_faces_np(self, obj_class=None):
        if obj_class == None:
            return self.smpl_m.faces
        else:
            return self.smpl_m.faces, [self.canonical_obj_meshes[c].faces for c in obj_class]

    def to(self, device):
        self.smpl_f = self.smpl_f.to(device)
        self.smpl_m = self.smpl_m.to(device)
        self.canonical_obj_keypoints_th = self.canonical_obj_keypoints_th.to(device)

        self.device = device

        return self

    @torch.no_grad()
    def get_meshes(self, output: TriDiModelOutput, obj_class, scale=1.0, sbj_gender=None):
        sbj_vertices, obj_vertices = self.get_meshes_th(output, obj_class, scale, sbj_gender)

        # create subject mesh
        sbj_vertices = sbj_vertices.cpu().numpy()
        sbj_faces = self.smpl_m.faces
        sbj_meshes = [trimesh.Trimesh(sbj_vertices[j], sbj_faces) for j in range(sbj_vertices.shape[0])]

        # create object mesh
        obj_meshes = []
        obj_vertices = obj_vertices.cpu().numpy()
        for j in range(obj_vertices.shape[0]):
            canonical_obj_mesh = self.canonical_obj_meshes[obj_class[j].cpu().item()]
            obj_meshes.append(
                trimesh.Trimesh(
                    obj_vertices[j, : len(canonical_obj_mesh.vertices)],
                    canonical_obj_mesh.faces
                )
            )

        return sbj_meshes, obj_meshes

    def get_smpl_th(self, params: Union[Dict, HHBatchData], sbj_gender=None):
        if isinstance(params, HHBatchData):
            #print("Getting SMPL from HHBatchData", params.to_string())
            B = len(params['sbj_shape'])
            sbj_gender = params.sbj_gender

            sbj_pose = params['sbj_pose'].reshape(B, -1, 6)
            sbj_pose = rotation_6d_to_matrix(sbj_pose).reshape(B, -1, 9).float()
            sbj_global = params['sbj_global'].reshape(B, 1, 6)
            sbj_global = rotation_6d_to_matrix(sbj_global).reshape(B, 1, 9).float()

            body_model_params = {
                "betas": params['sbj_shape'],
                "transl": params['sbj_c'],
                "global_orient": sbj_global,
                "body_pose": sbj_pose[:, :21],
                "left_hand_pose": sbj_pose[:, 21:36],
                "right_hand_pose": sbj_pose[:, 36:]
            }

            second_sbj_pose = params['second_sbj_pose'].reshape(B, -1, 6)
            second_sbj_pose = rotation_6d_to_matrix(second_sbj_pose).reshape(B, -1, 9).float()
            second_sbj_global = params['second_sbj_global'].reshape(B, 1, 6)
            second_sbj_global = rotation_6d_to_matrix(second_sbj_global).reshape(B, 1, 9).float()

            second_body_model_params = {
                "betas": params['second_sbj_shape'],
                "transl": params['second_sbj_c'],
                "global_orient": second_sbj_global,
                "body_pose": second_sbj_pose[:, :21],
                "left_hand_pose": second_sbj_pose[:, 21:36],
                "right_hand_pose": second_sbj_pose[:, 36:]
            }

        B = min(self.batch_size, len(body_model_params['betas']))
        body_model_params = {k: v.to(self.device) for k, v in body_model_params.items()}
        second_body_model_params = {k: v.to(self.device) for k, v in second_body_model_params.items()}

        # default to male
        # get smpl(-h) vertices
        sbj_output = self.smpl_m(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params)
        sbj_vertices = sbj_output['vertices']
        sbj_joints = sbj_output['joints']

        second_sbj_output = self.smpl_m(
            pose2rot=False, get_skin=True, return_full_pose=True, **second_body_model_params
        )
        second_sbj_vertices = second_sbj_output['vertices']
        second_sbj_joints = second_sbj_output['joints']

        return sbj_vertices, sbj_joints, second_sbj_vertices, second_sbj_joints
    
    def get_smpl_th_single(self, body_model_params: Union[Dict, HHBatchData], sbj_gender=None):
        B = min(self.batch_size, len(body_model_params['betas']))
        body_model_params = {k: v.to(self.device) for k, v in body_model_params.items()}
        # for k, v in body_model_params.items():
        #     print(f"  {k}: {v.shape}")

        # default to male
        # get smpl(-h) vertices
        sbj_output = self.smpl_m(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params)
        sbj_vertices = sbj_output['vertices']
        sbj_joints = sbj_output['joints']

        return sbj_vertices, sbj_joints

    @torch.no_grad()
    def get_smpl_np(self, body_model_params, sbj_gender=None):
        sbj_vertices, sbj_joints = self.get_smpl_th(body_model_params, sbj_gender)

        return sbj_vertices.cpu().numpy(), sbj_joints.cpu().numpy()

