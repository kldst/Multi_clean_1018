"""Small helpers shared by the evaluators. Metric names used everywhere:
  SMPL-22  SMPL J_regressor on SMPL-topology (smplx2smpl) vertices, body joints 0..21, root = joint 0
  H36M-14  J_regressor_h36m -> H36M_TO_J14 (the root differs per protocol, see each evaluator)
  PA-      after a similarity (Umeyama / Procrustes) alignment
"""
import pickle

import numpy as np

from multismpl.eval.protocols.common import procrustes
from multismpl.paths import PATHS


def load_smplx2smpl():
    with open(PATHS.smplx2smpl, 'rb') as h:
        return np.asarray(pickle.load(h)['matrix'], np.float32)


def load_smpl_jreg():
    with open(PATHS.smpl_neutral, 'rb') as h:
        r = pickle.load(h, encoding='latin1')['J_regressor']
    return np.asarray(r.todense() if hasattr(r, 'todense') else r, np.float32)


def smpl22_metrics(pv_smpl, gv_smpl, smpl_jreg):
    """SMPL-22 MPJPE / PA-MPJPE (mm) of one matched pair of SMPL-topology meshes (the MAMMA dance definition)."""
    ps, gs = (smpl_jreg @ pv_smpl)[:22], (smpl_jreg @ gv_smpl)[:22]
    ps, gs = ps - ps[0:1], gs - gs[0:1]
    return {'smpl22_mpjpe_mm': float(np.linalg.norm(ps - gs, axis=-1).mean() * 1000),
            'smpl22_pa_mpjpe_mm': float(np.linalg.norm(procrustes(ps, gs) - gs, axis=-1).mean() * 1000)}


def prediction_world(d, kind):
    """Inference npz (camera-0 gauge) or fit npz (world) -> pose (F,P,72) world orient, beta (F,P,10), pelvis (F,P,3)."""
    if kind == 'fit':
        return d['pose'], d['beta'], d['anchor']
    from multismpl.body.body_model import gauge_to_world
    pose, pelvis = gauge_to_world(d['pose'], d['trans'], d['E'][0])
    return pose, d['beta'], pelvis
