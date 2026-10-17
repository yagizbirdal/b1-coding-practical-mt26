"""Submarine dynamics, mission data, and closed-loop simulation."""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .control import PDController
from .terrain import generate_reference_and_limits

class Submarine:
    """Simple plant model with horizontal motion and controlled depth."""

    def __init__(self):

        self.mass = 1
        self.drag = 0.1
        self.actuator_gain = 1

        self.dt = 1 # Time step for discrete time simulation

        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1 # Constant velocity in x direction
        self.vel_y = 0


    def transition(self, action: float, disturbance: float):
        """Update the submarine's state based on the control action and disturbance."""

        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        force_y = -self.drag * self.vel_y + self.actuator_gain * (action + disturbance)
        acc_y = force_y / self.mass
        self.vel_y += acc_y * self.dt

    def get_depth(self) -> float:
        return self.pos_y
    
    def get_position(self) -> tuple:
        return self.pos_x, self.pos_y
    
    def reset_state(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1
        self.vel_y = 0
    
class Trajectory:
    """Store the submarine's position over time."""

    def __init__(self, position: np.ndarray):
        self.position = position  
        
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission: Mission):
        x_values = np.arange(len(mission.reference))
        min_depth = np.min(mission.cave_depth)
        max_height = np.max(mission.cave_height)

        plt.fill_between(x_values, mission.cave_height, mission.cave_depth, color='blue', alpha=0.3)
        plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.show()

@dataclass
class Mission:
    """Generate a mission with a reference path and cave boundaries from a CSV file or randomly."""

    reference: np.ndarray
    cave_height: np.ndarray
    cave_depth: np.ndarray

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_name: str):
        """Load a mission from CSV file"""

        mission_data = pd.read_csv(file_name)

        # Check that the required columns exist to avoid missing data
        for column in ['reference', 'cave_height', 'cave_depth']:
            if column not in mission_data.columns:
                raise ValueError(f"Missing column: {column}")

        reference = mission_data['reference'].values
        cave_height = mission_data['cave_height'].values
        cave_depth = mission_data['cave_depth'].values

        # Validate that the reference is always within cave boundaries to avoid impossible missions
        num_steps = len(reference)
        for time_step in range(num_steps):
            if not (cave_depth[time_step] <= reference[time_step] <= cave_height[time_step]):
                raise ValueError(f"Reference at time step {time_step} is out of bounds")
    
        return cls(reference, cave_height, cave_depth)


class ClosedLoop:
    """Simulate the closed-loop system of submarine and controller."""

    def __init__(self, plant: Submarine, controller: PDController):
        self.plant = plant
        self.controller = controller

    def simulate(self,  mission: Mission, disturbances: np.ndarray) -> Trajectory:

        T = len(mission.reference)
        if len(disturbances) < T:
            raise ValueError("Disturbances must be at least as long as mission duration")
        
        positions = np.zeros((T, 2))
        actions = np.zeros(T)
        self.plant.reset_state()
        self.controller.reset()

        for t in range(T):
            positions[t] = self.plant.get_position()
            observation_t = self.plant.get_depth()
            actions[t] = self.controller.step(mission.reference[t], observation_t)
            self.plant.transition(actions[t], disturbances[t])

        return Trajectory(positions)
        
    def simulate_with_random_disturbances(self, mission: Mission, variance: float = 0.5) -> Trajectory:
        disturbances = np.random.normal(0, variance, len(mission.reference))
        return self.simulate(mission, disturbances)
