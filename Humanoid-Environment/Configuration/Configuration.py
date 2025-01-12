class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Configuration(metaclass=SingletonMeta):
    def __init__(self):
        self.NUM_ITERATIONS = 1000 
        self.NUM_BINS_ACTION = 4
        self.UPDATE_POLICY_EVERY = 20
        self.N_EPISODES = 1000000
        self.N_EPISODES_TEST = 1000
