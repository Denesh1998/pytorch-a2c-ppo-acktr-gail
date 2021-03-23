from environments.worker import Worker
from environments.warehouse.robot import Robot

import multiprocessing as mp
import numpy as np
import random
import gym
from gym import spaces
import torch
class VectorizedEnvironment(object):
    """
    Creates multiple instances of an environment to run in parallel.
    Each of them contains a separate worker (actor) all of them following
    the same policy
    """
    ACTIONS = {0: 'UP',
               1: 'DOWN',
               2: 'LEFT',
               3: 'RIGHT'}
    
    def __init__(self, parameters, seed):
        print('cpu count', mp.cpu_count())
        if parameters['num_workers'] < mp.cpu_count():
            self.num_workers = parameters['num_workers']
        else:
            self.num_workers = mp.cpu_count()
        # Random seed needs to be set different for each worker (seed + worker_id). Otherwise multiprocessing takes 
        # the current system time, which is the same for all workers!
        self.workers = [Worker(parameters, worker_id, seed + worker_id) for worker_id in range(self.num_workers)]
        self.parameters = parameters
        self.env = parameters['env_type']

    def reset(self):
        """
        Resets each of the environment instances
        """
        for worker in self.workers:
            worker.child.send(('reset', None))
        output = {'obs': [], 'prev_action': []}
        for worker in self.workers:
            obs = worker.child.recv()
            if self.env == 'atari':
                stacked_obs = np.zeros((self.parameters['frame_height'],
                                        self.parameters['frame_width'],
                                        self.parameters['num_frames']))
                stacked_obs[:, :, 0] = obs[:, :, 0]
                obs = stacked_obs
            output['obs'].append(obs)
            output['prev_action'].append(-1)
        return output

    def step(self, actions, prev_stacked_obs):
        """
        Takes an action in each of the enviroment instances
        """
        for worker, action in zip(self.workers, actions):
            worker.child.send(('step', action))
        output = {'obs': [], 'reward': [], 'done': [], 'prev_action': [],
                  'info': []}
        i = 0
        for worker in self.workers:
            obs, reward, done, info = worker.child.recv()
            if self.parameters['flicker']:
                p = 0.5
                prob_flicker = random.uniform(0, 1)
                if prob_flicker > p:
                    obs = np.zeros_like(obs)
            if self.env == 'atari':
                new_stacked_obs = np.zeros((self.parameters['frame_height'],
                                            self.parameters['frame_width'],
                                            self.parameters['num_frames']))
                new_stacked_obs[:, :, 0] = obs[:, :, 0]
                new_stacked_obs[:, :, 1:] = prev_stacked_obs[i][:, :, :-1]
                obs = new_stacked_obs
            output['obs'].append(obs)
            output['reward'].append(reward)
            output['done'].append(done)
            output['info'].append(info)
            i += 1
        output['prev_action'] = actions
        return output

    
    @property
    def observation_space(self):
        """
        Returns the dimensions of the environment's action space
        """
        self.obs_dim = (self.parameters["frame_height"],
                                                    self.parameters["frame_width"],
                                                    self.parameters["num_frames"])
        high = 255* torch.ones(self.obs_dim)
        low = torch.zeros(self.obs_dim)
        observation_space = spaces.Box(0,255,(self.parameters["frame_height"],
                                                    self.parameters["frame_width"],
                                                    self.parameters["num_frames"]))
        print(observation_space.shape)
        return observation_space

    @property 
    def action_space(self):
        """
        Returns A gym dict containing the number of action choices for all the
        agents in the environment
        """
        n_actions = spaces.Discrete(len(self.ACTIONS))
        action_dict = {robot.get_id:n_actions for robot in self.robots}
        action_space = spaces.Dict(action_dict)
        action_space.n = 4
        return action_space

    def close(self):
        """
        Closes each of the threads in the multiprocess
        """
        for worker in self.workers:
            worker.child.send(('close', None))
