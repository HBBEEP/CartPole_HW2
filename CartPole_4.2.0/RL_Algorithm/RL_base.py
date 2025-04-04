import numpy as np
from collections import defaultdict
from enum import Enum
import os
import json
import torch


class ControlType(Enum):
    """
    Enum representing different control algorithms.
    """
    MONTE_CARLO = 1
    SARSA = 2
    Q_LEARNING = 3
    DOUBLE_Q_LEARNING = 4


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        control_type (ControlType): The type of control algorithm used.
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        control_type: ControlType,
        num_of_action: int,
        action_range: list,  # [min, max]
        discretize_state_weight: list,  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):
        self.control_type = control_type
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range
        self.discretize_state_weight = discretize_state_weight

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        if self.control_type == ControlType.MONTE_CARLO:
            self.obs_hist = []
            self.action_hist = []
            self.reward_hist = []
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            self.qa_values = defaultdict(lambda: np.zeros(self.num_of_action))
            self.qb_values = defaultdict(lambda: np.zeros(self.num_of_action))

    def discretize_state(self, obs: dict):
        """
        Discretize the observation state.

        Args:
            obs (dict): Observation dictionary containing policy states.

        Returns:
            Tuple[pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]: Discretized state.
        """

        # ========= put your code here =========#

        # self.discretize_state_weight
        policy_tensor = obs["policy"]

        if policy_tensor.is_cuda:
            policy_tensor = policy_tensor.cpu()

        policy_values = policy_tensor.numpy().flatten()

        pose_cart_bins = np.linspace(-1.0, 1.0, self.discretize_state_weight[0])   # Bins for pose_cart
        pose_pole_bins = np.linspace(-0.5, 0.5, self.discretize_state_weight[1])  # Bins for pose_pole
        vel_cart_bins = np.linspace(-float(np.deg2rad(24.0)), float(np.deg2rad(24.0)), self.discretize_state_weight[2])   # Bins for vel_cart
        vel_pole_bins = np.linspace(-float(np.deg2rad(24.0)), float(np.deg2rad(24.0)), self.discretize_state_weight[3])   # Bins for vel_pole

        pose_cart = np.digitize(policy_values[0], pose_cart_bins) - 1
        pose_pole = np.digitize(policy_values[1], pose_pole_bins) - 1
        vel_cart = np.digitize(policy_values[2], vel_cart_bins) - 1
        vel_pole = np.digitize(policy_values[3], vel_pole_bins) - 1

        return pose_cart, pose_pole, vel_cart, vel_pole
        # ======================================#

    def get_discretize_action(self, obs_dis) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            obs_dis (tuple): Discretized observation.

        Returns:
            int: Chosen discrete action index.
        """
        # ========= put your code here =========#
        prob = np.random.rand()

        if self.control_type == ControlType.MONTE_CARLO or self.control_type == ControlType.SARSA or self.control_type == ControlType.Q_LEARNING:
            if prob < self.epsilon:
                action = np.random.choice(self.num_of_action)

            else:
                action = np.argmax(self.q_values[obs_dis])

        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            if prob < self.epsilon:
                action = np.random.choice(self.num_of_action)
            else:
                action = np.argmax(self.qa_values[obs_dis] + self.qb_values[obs_dis])

        return action


        # ======================================#
    
    def mapping_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n]
            n (int): Number of discrete actions
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """
        # ========= put your code here =========#
        action_min = self.action_range[0]
        action_max = self.action_range[1]

        diff_action = action_max - action_min
        scaled_action = action_min + (action / (self.num_of_action - 1)) * diff_action

        return torch.tensor([[scaled_action]], dtype=torch.float32)
        # ======================================#

    def get_action(self, obs) -> torch.tensor:
        """
        Get action based on epsilon-greedy policy.

        Args:
            obs (dict): The observation state.

        Returns:
            torch.Tensor, int: Scaled action tensor and chosen action index.
        """
        obs_dis = self.discretize_state(obs)
        action_idx = self.get_discretize_action(obs_dis)
        action_tensor = self.mapping_action(action_idx)

        return action_tensor, action_idx
    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here =========#
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
        # ======================================#

    def save_q_value(self, path, filename):
        """
        Save the model parameters to a JSON file.

        Args:
            path (str): Path to save the model.
            filename (str): Name of the file.
        """
        # Convert tuple keys to strings
        try:
            q_values_str_keys = {str(k): v.tolist() for k, v in self.q_values.items()}
        except:
            q_values_str_keys = {str(k): v for k, v in self.q_values.items()}
        
        model_params = {
            'q_values': q_values_str_keys,
        }

        # Save additional parameters for Monte Carlo
        if self.control_type == ControlType.MONTE_CARLO:
            try:
                n_values_str_keys = {str(k): v.tolist() for k, v in self.n_values.items()}
            except:
                n_values_str_keys = {str(k): v for k, v in self.n_values.items()}
            model_params['n_values'] = n_values_str_keys

        # Save additional parameters for Double Q-Learning
        if self.control_type == ControlType.DOUBLE_Q_LEARNING:
            try:
                qa_values_str_keys = {str(k): v.tolist() for k, v in self.qa_values.items()}
                qb_values_str_keys = {str(k): v.tolist() for k, v in self.qb_values.items()}
            except:
                qa_values_str_keys = {str(k): v for k, v in self.qa_values.items()}
                qb_values_str_keys = {str(k): v for k, v in self.qb_values.items()}
            
            model_params['qa_values'] = qa_values_str_keys
            model_params['qb_values'] = qb_values_str_keys

        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as f:
            json.dump(model_params, f)

            
    def load_q_value(self, path, filename):
        """
        Load model parameters from a JSON file.

        Args:
            path (str): Path where the model is stored.
            filename (str): Name of the file.

        Returns:
            dict: The loaded Q-values.
        """
        full_path = os.path.join(path, filename)        
        with open(full_path, 'r') as file:
            data = json.load(file)
            data_q_values = data['q_values']
            
            for state, action_values in data_q_values.items():
                tuple_state = tuple(map(float, state.strip("()").split(', ')))
                self.q_values[tuple_state] = np.array(action_values)

            if self.control_type == ControlType.DOUBLE_Q_LEARNING:
                data_qa_values = data.get('qa_values', {})
                data_qb_values = data.get('qb_values', {})

                for state, action_values in data_qa_values.items():
                    tuple_state = tuple(map(float, state.strip("()").split(', ')))
                    self.qa_values[tuple_state] = np.array(action_values)

                for state, action_values in data_qb_values.items():
                    tuple_state = tuple(map(float, state.strip("()").split(', ')))
                    self.qb_values[tuple_state] = np.array(action_values)

            if self.control_type == ControlType.MONTE_CARLO:
                data_n_values = data.get('n_values', {})
                for state, n_values in data_n_values.items():
                    tuple_state = tuple(map(float, state.strip("()").split(', ')))
                    self.n_values[tuple_state] = np.array(n_values)

        return self.q_values

        # full_path = os.path.join(path, filename)        
        # with open(full_path, 'r') as file:
        #     data = json.load(file)
        #     data_q_values = data['q_values']
        #     for state, action_values in data_q_values.items():
        #         state = state.replace('(', '')
        #         state = state.replace(')', '')
        #         tuple_state = tuple(map(float, state.split(', ')))
        #         self.q_values[tuple_state] = action_values.copy()
        #         if self.control_type == ControlType.DOUBLE_Q_LEARNING:
        #             self.qa_values[tuple_state] = action_values.copy()
        #             self.qb_values[tuple_state] = action_values.copy()
        #     if self.control_type == ControlType.MONTE_CARLO:
        #         data_n_values = data['n_values']
        #         for state, n_values in data_n_values.items():
        #             state = state.replace('(', '')
        #             state = state.replace(')', '')
        #             tuple_state = tuple(map(float, state.split(', ')))
        #             self.n_values[tuple_state] = n_values.copy()
        #     return self.q_values

