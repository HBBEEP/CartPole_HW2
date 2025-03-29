from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Monte Carlo algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(
        self,
        obs, # state
        next_obs, # next state #### NOT USED
        action_idx,
        reward,
        
    ):
        """
        Update Q-values using Monte Carlo.

        This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
        """
        obs_dis = self.discretize_state(obs)
        # next_obs_dis = self.discretize_state(next_obs)

        self.update_hist(obs_dis, action_idx, reward)

        G = 0  # Initial Return
        for t in reversed(range(len(self.reward_hist))):
            state = self.obs_hist[t]
            action = self.action_hist[t]
            reward = self.reward_hist[t]
            
            G = reward + self.discount_factor * G  # Return
            if (state, action) not in zip(self.obs_hist[:t], self.action_hist[:t]):
                self.n_values[state][action] += 1
                weight_n = 1 / self.n_values[state][action] 
                self.q_values[state][action] += weight_n * (G - self.q_values[state][action])


    def update_hist(self, state, action, reward):
        self.obs_hist.append(state)
        self.action_hist.append(action)
        self.reward_hist.append(reward)

    def reset_hist(self):
        self.obs_hist.clear()
        self.action_hist.clear()
        self.reward_hist.clear()

    def save_model(self, path, filename):
        return super().save_q_value(path, filename)
    
    def load_model(self, path, filename):
        return super().load_q_value(path, filename)
