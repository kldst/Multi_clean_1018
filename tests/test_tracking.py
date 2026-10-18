"""Identity tracking on synthetic trajectories."""
import numpy as np

from multismpl.tto.tracking import track_candidates_joint, track_slots


def two_walkers(n=20):
    t = np.linspace(0, 1, n)
    a = np.stack([t, np.zeros(n), np.full(n, 4.)], 1)            # person A walks +x
    b = np.stack([1 - t, np.full(n, 0.8), np.full(n, 4.)], 1)    # person B walks -x, 0.8 m to the side
    return a, b


def test_slot_tracker_undoes_a_slot_swap():
    a, b = two_walkers()
    slots = np.stack([a, b], 1)
    slots[10:] = slots[10:, ::-1]                                # the network swaps its slot order at frame 10
    masks = np.zeros((len(a), 2, 1, 4, 4), np.uint8)
    perm, amb = track_slots(slots, masks)
    track0 = slots[np.arange(len(a)), perm[:, 0]]
    person = a if np.allclose(track0[0], a[0]) else b            # which person is called "track 0" is arbitrary ...
    np.testing.assert_allclose(track0, person)                   # ... but it must be the same person in every frame
    assert not amb.any()


def test_slot_tracker_single_person_is_identity():
    a, _ = two_walkers()
    perm, amb = track_slots(a[:, None], np.zeros((len(a), 1, 1, 4, 4), np.uint8))
    assert (perm == 0).all() and not amb.any()


def test_joint_tracker_ignores_a_phantom_and_follows_both_people():
    a, b = two_walkers()
    n = len(a)
    phantom = np.tile([[3., 0., 6.]], (n, 1))
    pos = np.stack([a, b, phantom], 1)                           # (F, C=3, 3)
    presence = np.tile([0.6, 0.7, 0.99], (n, 1))                 # the phantom is the most "present"
    support = np.tile([1.0, 1.0, 0.0], (n, 1))                   # ... but has no mask in any view
    cmasks = np.zeros((n, 3, 2, 8, 8), np.uint8)
    cmasks[:, 0, :, :4] = 255
    cmasks[:, 1, :, 4:] = 255
    choice = track_candidates_joint(pos, presence, 2, support, cmasks)
    assert (choice >= 0).all()
    assert not (choice == 2).any()                               # phantom never used
    assert len({tuple(sorted(c)) for c in choice}) == 1          # same pair every frame
    assert (choice[:, 0] == choice[0, 0]).all()                  # no identity switch
