# ARS AI
"""
Created on Sun Apr 19 21:50:57 2020

@developer: sepehr mohseni
"""

import numpy
import gym
from gym import wrappers
import pybullet_envs
import os


class HyperParameters():
    
    #these are the best values after several tests, if you are unhappy with results, change episode and learn values.
    def __init__(self):
        self.steps = 1000
        self.episode = 1000
        self.learn = 0.02
        self.noise = 0.35
        self.seed = 1
        self.directions = 60
        self.best = 16
        assert self.best <= self.directions
 
        self.environment = 'HalfCheetahBulletEnv-v0'
        #self.environment = 'Pendulum-v0'
        #self.environment = 'HopperBulletEnv-v0'
        

#implementing section 3.2 of original article
class Normaliser():
    
    def __init__(self, nb_inputs):
        self.n = numpy.zeros(nb_inputs)
        self.mean = numpy.zeros(nb_inputs)
        self.mean_diff = numpy.zeros(nb_inputs)
        self.var = numpy.zeros(nb_inputs)
        
    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
        
    def normalise(self, inputs):
        obs_mean = self.mean
        obs_std = numpy.sqrt(self.var)
        return (inputs - obs_mean) / obs_std
    

#implementing the brain
class AI():
    
    def __init__(self, input_size, output_size):
        self.theta = numpy.zeros((output_size, input_size))
             
#implementing section 3(our proposed algorithm) V2 of original article
    def evaluate(self, input, delta = None, direction = None):
        
        if direction is None:
            return self.theta.dot(input)
        
        elif direction == "positive":
            return (self.theta + hyperparams.noise * delta).dot(input)
        
        else:
            return (self.theta - hyperparams.noise * delta).dot(input)
    
    def sample_perturbation(self):
        return [numpy.random.randn(*self.theta.shape) 
                for i in range(hyperparams.directions)]
        
    #step 7 of section 3
    def update(self, rollouts, sigma_r):
        #finite difference method of equation
        step = numpy.zeros(self.theta.shape)
        
        for r_positives, r_negatives, d in rollouts:
            step += (r_positives - r_negatives) * d
        
        self.theta += hyperparams.learn / (hyperparams.best * sigma_r) * step
        
        
def explore(env, normaliser, ai, direction = None, delta = None):
    state = env.reset()
    done = False
    num_of_plays = 0.
    sum_rewards = 0
    
    while done != True and num_of_plays < hyperparams.episode:
        normaliser.observe(state)
        state = normaliser.normalise(state)
        action = ai.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_of_plays += 1
        
    return sum_rewards


#training brain. step 3 of section 3 of article
def train(env, ai, normaliser, hyperparams):
    
    for step in range(hyperparams.steps):
        deltas = ai.sample_perturbation()
        positive_rewards = [0] * hyperparams.directions
        negative_rewards = [0] * hyperparams.directions
        
        for k in range(hyperparams.directions):
            positive_rewards[k] = explore(env, normaliser, ai, direction = "positive", delta = deltas[k])
          
        for k in range(hyperparams.directions):
            negative_rewards[k] = explore(env, normaliser, ai, direction = "negative", delta = deltas[k])
    
        all_rewards_list = numpy.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards_list.std()
        
        #step 6
        scores = {k:max(r_pos, r_neg) for k,(r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x])[0:hyperparams.best]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        #updating AI
        ai.update(rollouts, sigma_r)
        
        #obtaining the final result
        reward_evaluation = explore(env, normaliser, ai)
        print('Try: ', step, 'Result: ', reward_evaluation)
        
    
#execution    
def mkdir(base, name):
    
    path = os.path.join(base, name)
    
    if not os.path.exists(path):
        os.makedirs(path)
    return path

work_dir = mkdir('ai', 'video')
watching_dir = mkdir(work_dir, 'watching')


hyperparams = HyperParameters()
numpy.random.seed(hyperparams.seed)
env = gym.make(hyperparams.environment)

env = wrappers.Monitor(env, watching_dir, force = True)

nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]
ai = AI(nb_inputs, nb_outputs)

normaliser = Normaliser(nb_inputs)

train(env, ai, normaliser, hyperparams)
