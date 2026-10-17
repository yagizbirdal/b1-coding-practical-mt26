from __future__ import annotations


class PDController:
    """Discrete-time PD depth controller.

    Implements u[t] = kp * e[t] + kd * (e[t] - e[t-1]), with e[t] = r[t] - y[t]
    and e[-1] = 0. The previous error is stored on the instance, so the
    controller must be called once per time step, in order.
    """

    def __init__(self, kp: float = 0.15, kd: float = 0.6):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0.0  # e[-1] = 0

    def reset(self) -> None:
        """Clear the stored error so a new simulation starts from e[-1] = 0."""
        self.prev_error = 0.0

    def step(self, reference: float, measurement: float) -> float:
        """Return the control action u[t] for one time step.

        Args:
            reference: desired depth r[t].
            measurement: measured depth y[t].
        """
        error = reference - measurement
        action = self.kp * error + self.kd * (error - self.prev_error)
        self.prev_error = error
        return action
