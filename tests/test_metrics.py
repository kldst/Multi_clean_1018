"""Basic properties of the evaluation protocols (synthetic meshes, random regressors)."""
import numpy as np

from multismpl.eval.protocols import chi3d, harmony4d, mamma
from multismpl.eval.protocols.common import match_greedy_pve, procrustes, top2_nms3d

rng = np.random.default_rng(0)
SMPL_JREG = rng.random((24, 6890)).astype(np.float32)
SMPL_JREG /= SMPL_JREG.sum(1, keepdims=True)
H36M_JREG = rng.random((17, 6890)).astype(np.float32)
H36M_JREG /= H36M_JREG.sum(1, keepdims=True)
MESH = rng.normal(size=(6890, 3)).astype(np.float64) * 0.3


def rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_procrustes_undoes_a_similarity_transform():
    X = rng.normal(size=(14, 3))
    Y = 1.3 * X @ rot(40).T + np.array([0.5, -1., 2.])
    np.testing.assert_allclose(procrustes(X, Y), Y, atol=1e-9)


def test_identical_meshes_score_zero():
    m = mamma.fair_regressed_metrics(MESH, MESH, SMPL_JREG, H36M_JREG)
    assert max(m.values()) < 1e-9


def test_rooted_metrics_ignore_translation_but_transl_does_not():
    m = mamma.fair_regressed_metrics(MESH + [0.1, 0, 0], MESH, SMPL_JREG, H36M_JREG)
    assert m['mpjpe22_mm'] < 1e-4 and m['mpjpe14_mm'] < 1e-4                  # float32 regressors: ~1e-6 mm round-off
    assert abs(m['transl_mm'] - 100) < 1e-3
    pel = (H36M_JREG @ MESH)[0:1]
    h = harmony4d.pair_metrics(MESH + [0.1, 0, 0], MESH, pel + [0.1, 0, 0], pel, H36M_JREG)
    assert h['mpjpe_mm'] < 1e-4 and abs(h['transl_mm'] - 100) < 1e-3


def test_pa_metrics_ignore_rotation():
    c = MESH.mean(0)
    m = mamma.fair_regressed_metrics((MESH - c) @ rot(25).T + c, MESH, SMPL_JREG, H36M_JREG)
    assert m['mpjpe22_mm'] > 1 and m['pa_mpjpe22_mm'] < 1e-6


def test_top2_nms3d_skips_duplicates_within_25cm():
    prob = np.array([0.9, 0.8, 0.7])
    roots = np.array([[0., 0, 0], [0.1, 0, 0], [1., 0, 0]])     # slot 1 duplicates slot 0
    assert top2_nms3d(prob, roots) == [0, 2]


def test_greedy_pve_gate_rejects_far_pairs():
    gt = [MESH, MESH + [2., 0, 0]]
    pred = [MESH + [0.01, 0, 0], MESH + [5., 0, 0]]               # second prediction is 3 m from anyone
    matches, fp = match_greedy_pve(pred, gt, 500.)
    assert matches == [(0, 0)] and fp == 1
    matches, fp = match_greedy_pve(pred, gt, np.inf)
    assert sorted(matches) == [(0, 0), (1, 1)] and fp == 0


def test_chi3d_aggregate_recall():
    pairs = [chi3d.pair_metrics(MESH, MESH, MESH[:1], MESH[:1], MESH[:22], MESH[:22], H36M_JREG)] * 3
    out = chi3d.aggregate(pairs, gt_person_frames=4, false_positives=1)
    assert out['recall'] == 75 and out['precision'] == 75 and out['mpjpe_mm'] == 0
