from Configuration import Configuration as Conf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class ProximalPolicyOptimization:
    def __init__(self):
        self.learning_rate = 3e-4
        self.n_steps = 2048  
        self.batch_size = 64      
        self.n_epochs = 10         
        self.gamma = 0.99         
        self.gae_lambda = 0.95     
        self.clip_range = 0.2      
        self.verbose = 1
        self.conf = Conf.Configuration()
        self.model = None
        self.data_directory_path = "data/ProximalPolicyOptimization/"

    def initAgent(self, env):
        '''Initialize the agent with the environment'''
        vec_env = DummyVecEnv([lambda: env])
        self.model = PPO(
            policy="MlpPolicy",  
            env=vec_env,
            learning_rate=self.learning_rate,  
            n_steps=self.n_steps,       
            batch_size=self.batch_size,      
            n_epochs=self.n_epochs,         
            gamma=self.gamma,         
            gae_lambda=self.gae_lambda,     
            clip_range=self.clip_range,      
            verbose=self.verbose            
        )

    def train(self, env):
        '''Train the agent based on the environment'''
        if self.model is None:
            self.initAgent(env)
        self.model.learn(total_timesteps=self.conf.N_EPISODES)

    def choose_action(self, state):
        '''Choose an action based on the state'''
        action, _states = self.model.predict(state, deterministic=True)
        return action

    def save_model(self, file_name):
        '''Save the model to the specified file name'''
        self.model.save(self.data_directory_path + file_name)

    def load_model(self, file_name, env):
        '''Load the model from the specified file name'''
        self.model = PPO.load(self.data_directory_path + file_name)
        self.model.set_env(env)

