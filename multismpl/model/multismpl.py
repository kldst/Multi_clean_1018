"""The multi-view, multi-person SMPL-X network (a VGGT derivative).

  aggregator                       VGGT aggregator (DINOv2 ViT-L patch embed + 24 alternating frame / global
                                   attention blocks); runs once per time step on the V views of that step
  smpl_multi_query_trans_rot_head  temporal multi-person SMPL head: P person queries cross-attend the patch tokens
                                   of a T-frame window -> per slot: presence, SMPL-X pose / beta, pelvis translation
                                   (camera-0 gauge, metres), person tokens
  person_mask_head                 DPT mask head: per person token and view, a full-resolution mask logit map
  camera_head                      VGGT camera head (kept so released checkpoints load strictly; the pipeline
                                   uses calibrated cameras and drops it after loading)

Attribute names match the training code, so checkpoints load with strict=True. The configuration of the released
model is configs/model_nano4.yaml.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import yaml

from multismpl.model.aggregator import Aggregator
from multismpl.model.heads.camera_head import CameraHead
from multismpl.model.heads.person_mask_head import PersonMaskDPTHead
from multismpl.model.heads.smpl_temporal_head import SMPLMultiQueryTemporalRelConfig, SMPLMultiQueryTransRotTemporalRelHead


class MultiSMPL(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, patch_embed='dinov2_vitl14_reg',
                 intermediate_layer_idx=(4, 11, 17, 23), frames_chunk_size=1, num_people=20,
                 temporal_rel_num_buckets=64, temporal_rel_max_distance=89, temporal_head_max_T=90,
                 mask_embed_dim=128, mask_down_ratio=1):
        super().__init__()
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                                     num_heads=num_heads, patch_embed=patch_embed)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.smpl_multi_query_trans_rot_head = SMPLMultiQueryTransRotTemporalRelHead(
            dim_in=2 * embed_dim, num_people=num_people,
            smpl_cfg=SMPLMultiQueryTemporalRelConfig(max_T=int(temporal_head_max_T), rel_num_buckets=int(temporal_rel_num_buckets),
                                                     rel_max_distance=int(temporal_rel_max_distance)))
        self.person_mask_head = PersonMaskDPTHead(dim_in=2 * embed_dim, query_dim=embed_dim, embed_dim=int(mask_embed_dim),
                                                  intermediate_layer_idx=list(intermediate_layer_idx),
                                                  down_ratio=int(mask_down_ratio), patch_size=patch_size,
                                                  frames_chunk_size=frames_chunk_size)

    # the trainer keeps the clip axis for temporal models (multismpl.train.trainer.forward_pass)
    use_temporal_smpl_head = True

    def forward(self, images: torch.Tensor, smpl_inputs: Optional[Dict[str, torch.Tensor]] = None):
        """images (B, T*V, 3, H, W) in [0,1], time-major; smpl_inputs: temporal_num_frames (B,), views_per_frame (B,).
        The aggregator sees one time step at a time (T folded into the batch); the SMPL head fuses the T steps.
        Returns the SMPL head outputs + person_mask_logits, and pose_enc / pose_enc_list when the camera head is kept
        (training: the camera loss uses every refinement iteration). In eval mode the input images are returned too."""
        smpl_inputs = smpl_inputs or {}
        if images.dim() == 4:
            images = images.unsqueeze(0)
        T = int(smpl_inputs['temporal_num_frames'].reshape(-1)[0])
        V = int(smpl_inputs['views_per_frame'].reshape(-1)[0])
        B, S = images.shape[:2]
        if S != T * V:
            raise ValueError(f'images have S={S}, expected T*V={T * V}')
        per_step = images.reshape(B, T, V, *images.shape[2:]).reshape(B * T, V, *images.shape[2:])
        tokens, patch_start_idx, _ = self.aggregator(per_step)
        pred = {}
        with torch.autocast(images.device.type, enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(tokens)
                pose_enc_list = pose_enc_list if isinstance(pose_enc_list, list) else [pose_enc_list]
                pred['pose_enc'], pred['pose_enc_list'] = pose_enc_list[-1], pose_enc_list
            pred.update(self.smpl_multi_query_trans_rot_head(tokens, patch_start_idx=patch_start_idx, smpl_inputs=smpl_inputs))
            pred['person_mask_logits'] = self.person_mask_head(pred['person_tokens'], tokens, images=per_step,
                                                               patch_start_idx=patch_start_idx)
        if not self.training:
            pred['images'] = images
        return pred


def build_model(config_path):
    with open(config_path) as h:
        return MultiSMPL(**yaml.safe_load(h))


def load_model(config_path, checkpoint, device, keep_camera_head=False):
    """Strict checkpoint load. The camera head is dropped by default (the pipeline uses calibrated cameras)."""
    model = build_model(config_path)
    ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(ck.get('model', ck), strict=True)
    del ck
    if not keep_camera_head:
        model.camera_head = None
    return model.eval().to(device)
