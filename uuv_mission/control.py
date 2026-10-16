from __future__ import annotations


class PDController:
    def __init__(self, kp: float = 0.15, kd: float = 0.6):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0

    def step(self, reference: float, measurement: float) -> float:
        raise NotImplementedError
