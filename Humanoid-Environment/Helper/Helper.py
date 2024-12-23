import numpy as np

class Helper:
    def __init__(self):
        pass
    
    def discretize_state(self, state):
        """Discretize continuous state as hashable state."""
        return tuple(state)