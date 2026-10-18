"""Running the trained model (VGGT aggregator + temporal SMPL head + DPT person-mask head) for step 1.

The three inference scripts differ only in how they read their dataset; the model calls live here.
"""
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from multismpl.data.base_dataset import BaseDataset
from multismpl.model.multismpl import load_model  # noqa: F401  (re-exported for the inference scripts)

MASK_SIZE = 259


def head_meta(n_views, device, T=3):
    return {'temporal_num_frames': torch.tensor([T], device=device), 'views_per_frame': torch.tensor([n_views], device=device)}


def image_preprocessor():
    """The training-time crop / resize to 518 (process_one_image), without augmentation."""
    pre = BaseDataset(SimpleNamespace(img_size=518, patch_size=14, augs=SimpleNamespace(scales=None), rescale=True,
                                      rescale_aug=False, landscape_check=False, emit_dense_geometry=False))
    pre.training = False
    pre.debug = False
    return pre


class CachedTemporalHead:
    """Causal T = 3 evaluation contract: the aggregator runs on the current frame only, its last-layer tokens are
    cached, and the temporal SMPL head sees [t-2, t-1, t] (the first frames repeat the oldest available)."""

    def __init__(self, model, meta):
        self.model, self.meta = model, meta
        self.history = []

    def reset(self):
        self.history = []

    def __call__(self, images):
        """images (V,3,518,518) float in [0,1] -> (pred, tokens, patch_start_idx)."""
        dev_type = images.device.type
        with torch.autocast(dev_type, dtype=torch.bfloat16):
            tokens, patch, _ = self.model.aggregator(images.unsqueeze(0))
            current = tokens[-1]
            h = self.history
            window = [h[0] if h else current] * (2 - len(h)) + h + [current]
            with torch.autocast(dev_type, enabled=False):
                pred = self.model.smpl_multi_query_trans_rot_head([torch.cat(window, dim=0)], patch_start_idx=patch,
                                                                  smpl_inputs=self.meta)
            self.history = (h + [current.detach()])[-2:]
        return pred, tokens, patch


def slot_outputs(pred):
    """Per query slot (Q of them): presence prob (Q,), pelvis (Q,3) camera-0 gauge, pose (Q,72), beta (Q,10)."""
    prob = pred['smpl_presence_logits'][-1].float().sigmoid().cpu().numpy().reshape(-1)
    trans = pred['mesh_translate'][-1].float().cpu().numpy().reshape(-1, 3)
    pose = pred['smpl_pose'][-1].float().cpu().numpy()
    beta = pred['smpl_beta'][-1].float().cpu().numpy()
    return prob, trans, pose, beta


def person_masks(model, pred, tokens, images, patch, slots):
    """DPT mask head for the given slots on the current frame -> (k,V,259,259) uint8 = sigmoid * 255.
    tokens: aggregator tokens of the current frame; images (1,V,3,518,518)."""
    with torch.autocast(images.device.type, enabled=False):
        logits = model.person_mask_head(pred['person_tokens'][-1:, slots], tokens, images=images, patch_start_idx=patch)
    m = logits[0].float().sigmoid().transpose(0, 1)                                   # (k,V,H,W)
    m = F.interpolate(m, size=(MASK_SIZE, MASK_SIZE), mode='bilinear', align_corners=False)
    return (m * 255).round().byte().cpu().numpy()


def rank_candidates_by_mask(masks_all, prob, selected, C):
    """C candidates for the v9 tracker: the evaluator's selected slots first, then the other slots ordered by
    (mask visible in >= half the views, presence), skipping duplicates (mean-over-views mask IoU > 0.5 with a kept
    one). A real but under-confident person (presence 0.05, full mask) then outranks phantom slots (presence 0.1,
    empty mask). masks_all (Q,V,G,G) uint8. Returns C slot indices (padded by repeating the last)."""
    fg = masks_all >= 128
    sup = (fg.sum((-1, -2)) > 300).mean(-1)
    flat = fg.reshape(len(fg), fg.shape[1], -1)

    def iou(i, j):
        inter = (flat[i] & flat[j]).sum(-1)
        union = (flat[i] | flat[j]).sum(-1)
        return float(np.where(union > 0, inter / np.maximum(union, 1), 0).mean())
    csel = list(selected)
    for i in sorted(range(len(prob)), key=lambda i: (-float(sup[i] >= 0.5), -float(prob[i]))):
        if len(csel) == C:
            break
        if i not in csel and all(iou(i, j) <= 0.5 for j in csel):
            csel.append(int(i))
    return (csel + [csel[-1]] * C)[:C]
