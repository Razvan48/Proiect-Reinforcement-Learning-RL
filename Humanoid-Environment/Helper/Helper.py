import numpy as np
from Configuration import Configuration as Conf
class Helper:
    def __init__(self):
        self.conf = Conf.Configuration()
    
    def discretize_state(self, state):
        """Discretize continuous state as hashable state."""
        state = np.round(state, 1)
        state = np.clip(state, -10, 10)
        return tuple(state)
    
    def discretize_action(self, action):
        """Discretize continuous action into numbered bins in the range [-0.4, 0.4]."""
        num_bins = self.conf.NUM_BINS_ACTION  
        action_min, action_max = -0.4, 0.4  

        bins = np.linspace(action_min, action_max, num_bins + 1)  
        discretized_action = np.digitize(action, bins) - 1

        return tuple(discretized_action)
    
    def reverse_discretize_action(self, discretized_action):
        """Reverse the discretized action to continuous action."""
        num_bins = self.conf.NUM_BINS_ACTION 
        action_min, action_max = -0.4, 0.4

        bin_width = (action_max - action_min) / num_bins
        bin_centers = np.linspace(action_min + bin_width / 2, action_max - bin_width / 2, num_bins)

        continuous_action = [bin_centers[i] for i in discretized_action]

        return tuple(continuous_action)
    
    def save_model(self, policy, file_name):
        """Save the policy to the specified file name."""
        with open(file_name, "w") as f:
            for state, action in policy.items():
                f.write(f"{state}: {action}\n")

    def load_model(self, file_name):
        """Load the policy from the specified file name."""
        policy = {}
        with open(file_name, "r") as f:
            for line in f:
                state, action = line.split(": ")
                cleaned_action = action[1:len(action) - 1].replace(')', '').replace(' ', '')
                policy[state] = tuple(map(int, cleaned_action.split(',')))
        return policy