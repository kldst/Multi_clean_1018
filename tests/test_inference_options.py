"""Per-sequence options of the MAMMA inference (no model needed)."""
from types import SimpleNamespace

import pytest

from multismpl.inference.mamma import people_in_name, resolve_counts


def test_people_count_is_read_from_the_markerless_name():
    assert people_in_name('260216_MultiMama_3_assist_walk_101100_1') == 3
    assert people_in_name('260216_MultiMama_4_hands_stack_huddle_011110_1') == 4
    with pytest.raises(SystemExit):
        people_in_name('250225_WestCoastSwing_Improv22_03675_03676_1')


def test_auto_counts_and_explicit_counts():
    auto = resolve_counts(SimpleNamespace(people='auto', candidates='auto'), '260216_MultiMama_3_conga_line_100110_1')
    assert (auto.people, auto.candidates) == (3, 7)                 # people + 4
    small = resolve_counts(SimpleNamespace(people='1', candidates='auto'), 'x')
    assert (small.people, small.candidates) == (1, 6)                # at least 6
    fixed = resolve_counts(SimpleNamespace(people='2', candidates='0'), 'x')
    assert (fixed.people, fixed.candidates) == (2, 0)                # dance defaults: no candidates
