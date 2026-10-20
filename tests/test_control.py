import pytest

from uuv_mission.control import PDController


def test_pd_sequence():
    c = PDController()
    out = [c.step(1, y) for y in (0, 0.5, 0.8)]
    assert out == pytest.approx([0.75, -0.225, -0.15])


def test_reset():
    c = PDController()
    c.step(1, 0)
    c.reset()
    assert c.step(1, 0) == pytest.approx(0.75)
