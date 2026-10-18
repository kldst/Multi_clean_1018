"""MAMMA dance / singles protocol (from eval/mamma_eval_dance/eval_mamma_dance_comparison.py and its singles copy
eval_mamma_singles.py; function bodies unchanged). Both prediction and GT SMPL-X meshes are mapped to the 6890-vertex
SMPL topology; SMPL-22 (SMPL J_regressor, root = joint 0) and H36M-14 (J_regressor_h36m, root = H36M pelvis) are
regressed with the same operators on both sides.

GT is decoded with each person's gendered SMPL-X model (singles GT is female / male; the dance GT is all neutral, for
which this is exactly the dance evaluator's neutral decode).
"""
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from smplx.lbs import lbs

from multismpl.body.smplx_body import _get_smplx_model, set_smplx_model_root
from multismpl.eval.protocols.common import H36M_TO_J14, procrustes

N_BODY = 22
GT_CHUNK = 64
GENDER_TENSORS = {}


def fair_regressed_metrics(pv_smpl, gv_smpl, smpl_j_regressor, h36m_j_regressor):
    """Score both joint sets with identical operators and roots on both sides."""
    ps, gs = smpl_j_regressor @ pv_smpl, smpl_j_regressor @ gv_smpl
    ph, gh = h36m_j_regressor @ pv_smpl, h36m_j_regressor @ gv_smpl
    ps22, gs22 = ps[:N_BODY], gs[:N_BODY]
    ps22 = ps22 - ps22[0:1]
    gs22 = gs22 - gs22[0:1]
    ph14 = (ph - ph[0:1])[H36M_TO_J14]
    gh14 = (gh - gh[0:1])[H36M_TO_J14]
    return dict(
        mpjpe22_mm=float(np.linalg.norm(ps22 - gs22, axis=-1).mean() * 1000),
        pa_mpjpe22_mm=float(np.linalg.norm(procrustes(ps22, gs22) - gs22, axis=-1).mean() * 1000),
        mpjpe14_mm=float(np.linalg.norm(ph14 - gh14, axis=-1).mean() * 1000),
        pa_mpjpe14_mm=float(np.linalg.norm(procrustes(ph14, gh14) - gh14, axis=-1).mean() * 1000),
        transl_mm=float(np.linalg.norm(ph[0] - gh[0]) * 1000),
    )


def load_body_tensors(model_path):
    with open(model_path, 'rb') as h:
        raw = pickle.load(h, encoding='latin1')

    def np_(v):
        if hasattr(v, 'r'):
            return np.asarray(v.r, dtype=np.float64)
        if hasattr(v, 'todense'):
            return np.asarray(v.todense(), dtype=np.float64)
        return np.asarray(v, dtype=np.float64)

    parents = torch.tensor(np_(raw['kintree_table'])[0].astype(np.int64))
    parents[0] = -1
    t = lambda k: torch.tensor(np_(raw[k]), dtype=torch.float32)  # noqa: E731
    return dict(posedirs=t('posedirs').reshape(-1, 486).T.contiguous(),
                J_regressor=t('J_regressor'), parents=parents,
                lbs_weights=t('weights'), shapedirs=t('shapedirs')[:, :, :300])


def decode_gt(seq_dir, tensors, n_frames, smplx2smpl=None):
    """Native SMPL-X ground truth in world coordinates (personalised v_template, per-person gender).
    Returns joints (F,P,22,3), SMPL-topology vertices (F,P,6890,3) when `smplx2smpl` is given, n_frames, n_people."""
    meta = np.load(seq_dir / 'gt' / 'global.npz', allow_pickle=True)
    n_frames = min(n_frames, int(meta['frames_len']))
    n_people = int(meta['people_len'])
    per_person, per_person_v = [], []
    with torch.inference_mode():
        for p in range(n_people):
            betas = torch.tensor(meta['shape'][p:p + 1], dtype=torch.float32)
            v_template = torch.tensor(meta['v_template'][p], dtype=torch.float32)
            shift = meta['pose_trans_world'][:n_frames, p][:, None, :]
            chunks, chunks_v = [], []
            for b in range(0, n_frames, GT_CHUNK):
                pose = torch.tensor(meta['pose_world'][b:min(n_frames, b + GT_CHUNK), p], dtype=torch.float32)
                v, j = lbs(betas=betas.expand(len(pose), -1), pose=pose, v_template=v_template,
                           **GENDER_TENSORS.get(str(meta['gender'][p]), tensors))
                chunks.append(j[:, :N_BODY].numpy())
                if smplx2smpl is not None:
                    chunks_v.append(torch.einsum('sv,fvc->fsc', smplx2smpl, v.to(smplx2smpl.device, torch.float32)).cpu().numpy())
            per_person.append(np.concatenate(chunks) + shift)
            if smplx2smpl is not None:
                per_person_v.append(np.concatenate(chunks_v) + shift)
    joints = np.stack(per_person, axis=1).astype(np.float32)
    verts = np.stack(per_person_v, axis=1).astype(np.float32) if smplx2smpl is not None else None
    return joints, verts, n_frames, n_people


def aggregate(pairs, gt_people, selected_people):
    pairs = [p for p in pairs if p]
    keys = ['mpjpe22_mm', 'mpjpe24_mm', 'pa_mpjpe22_mm', 'transl_mm', 'mpjpe14_mm', 'pa_mpjpe14_mm', 'pve_mm', 'pa_pve_mm']
    out = {k: (float(np.mean([p[k] for p in pairs])) if pairs and k in pairs[0] else None) for k in keys}
    matched = len(pairs)
    recall = matched / gt_people if gt_people else 0.0
    precision = matched / selected_people if selected_people else 0.0
    out.update(gt_people=gt_people, selected_people=selected_people, matched_people=matched,
               missed=gt_people - matched, recall=100 * recall, precision=100 * precision,
               f1=100 * 2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return out


def load_eval_assets(smplx_model_dir, smplx2smpl_path, j_regressor_h36m_path, smpl_model_path, device):
    """Body models and regressors shared by every sequence."""
    set_smplx_model_root(smplx_model_dir)
    body = _get_smplx_model(torch.device('cpu'), 'neutral')
    gt_tensors = load_body_tensors(Path(smplx_model_dir) / 'neutral' / 'model.pkl')
    GENDER_TENSORS.update({g: load_body_tensors(Path(smplx_model_dir) / g / 'model.pkl') for g in ('neutral', 'female', 'male')})
    with open(smplx2smpl_path, 'rb') as h:
        smplx2smpl = torch.tensor(np.asarray(pickle.load(h)['matrix'], dtype=np.float32), device=device)
    j_regressor = np.load(j_regressor_h36m_path).astype(np.float32)
    with open(smpl_model_path, 'rb') as h:
        smpl_raw = pickle.load(h, encoding='latin1')
    smpl_j_regressor = smpl_raw['J_regressor']
    smpl_j_regressor = np.asarray(smpl_j_regressor.todense() if hasattr(smpl_j_regressor, 'todense') else smpl_j_regressor,
                                  dtype=np.float32)
    return SimpleNamespace(body=body, gt_tensors=gt_tensors, smplx2smpl=smplx2smpl,
                           j_regressor=j_regressor, smpl_j_regressor=smpl_j_regressor)


def load_sequence_gt(seq_dir, assets, max_frames=0):
    """(gt_j, gt_v_smpl, n_frames, n_people) in world space."""
    n_frames = int(np.load(seq_dir / 'gt/global.npz')['frames_len'])
    if max_frames:
        n_frames = min(n_frames, max_frames)
    return decode_gt(seq_dir, assets.gt_tensors, n_frames, assets.smplx2smpl)
