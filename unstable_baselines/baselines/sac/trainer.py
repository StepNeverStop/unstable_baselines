from unstable_baselines.common.trainer import BaseTrainer
import numpy as np
import os
from tqdm import trange


def euclidean_distance(x,y):
    diffs = [(a-b)**2 for a,b in zip(x,y)]
    return np.sqrt(sum(diffs))

def manhattan_distance(x,y):
    diffs = [np.abs(a-b) for a,b in zip(x,y)]
    return sum(diffs)

class SACTrainer(BaseTrainer):
    def __init__(self, 
            agent, 
            opponent_agent,
            train_env, 
            eval_env, 
            buffer,  
            batch_size,
            max_env_steps,
            start_timestep,
            random_policy_timestep, 
            agent_idx,
            load_dir="",
            **kwargs):
        super(SACTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.agent = agent
        self.opponent_agent = opponent_agent
        self.buffer = buffer
        #hyperparameters
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.start_timestep = start_timestep
        self.random_policy_timestep = random_policy_timestep
        self.agent_idx = agent_idx
        if load_dir != "" and os.path.exists(load_dir):
            self.agent.load(load_dir)

    def train(self):
        train_traj_returns = [0]
        train_traj_lengths = [0]
        match_results = []
        tot_env_steps = 0
        traj_return = 0
        traj_length = 0
        done = False
        obs = self.train_env.reset()
        for env_step in trange(self.max_env_steps): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            self.pre_iter()
            log_infos = {}
            obs_agent = np.array(obs[self.agent_idx]['obs']).flatten() 
            obs_opponent = np.array(obs[1 - self.agent_idx]['obs']).flatten()
            if tot_env_steps < self.random_policy_timestep:
                action = self.train_env.action_space.sample()
            else:
                agent_action = self.agent.select_action(obs_agent)['action']
                opponent_action = self.opponent_agent.select_action(obs_opponent)['action']
            if self.agent_idx == 0:
                joint_action = [agent_action, opponent_action]
            else:
                joint_action = [opponent_action, agent_action]
            next_obs, joint_reward, done, _ = self.train_env.step(action)
            shaped_reward = self.shape_reward(joint_reward, done)
            traj_length += 1
            if traj_length >= self.max_trajectory_length:
                done = False
            #add transition to buffer if and only
            if self.train_env.env_core.current_team == self.agent_idx:
                traj_return += joint_reward[self.agent_idx]
                self.buffer.add_transition(obs_agent, joint_action[self.agent_idx], next_obs[self.agent_idx]['obs'], joint_reward[self.agent_idx], float(done))


            obs = next_obs
            if done or traj_length >= self.max_trajectory_length:
                winned = 1 if joint_reward[self.agent_idx]>joint_reward[1-self.agent_idx] else 0
                match_results.append(winned)
                obs = self.train_env.reset()
                train_traj_returns.append(traj_return)
                train_traj_lengths.append(traj_length)
                traj_length = 0
                traj_return = 0
            log_infos['performance/train_return'] = train_traj_returns[-1]
            log_infos['performance/train_length'] = train_traj_lengths[-1]
            tot_env_steps += 1
            if tot_env_steps < self.start_timestep:
                continue
    
            data_batch = self.buffer.sample(self.batch_size)
            #flatten obs
            data_batch['obs'] = data_batch['obs'].reshape(self.batch_size, -1)
            data_batch['next_obs'] = data_batch['next_obs'].reshape(self.batch_size, -1)
            train_agent_log_infos = self.agent.update(data_batch)
            log_infos.update(train_agent_log_infos)
           
            self.post_iter(log_infos, tot_env_steps)



        def shape_reward(self, raw_reward, done):
            #info to calculate reward
            agent_pos = self.train_env.env_core.agent_pos[-1]
            if done:
                shaped_reward = [-1, -1]
            else:
                if raw_reward[0] != raw_reward[1]:
                    shaped_reward = [raw_reward[0]-100, raw_reward[1]] if raw_reward[0]<raw_reward[1] else [raw_reward[0], raw_reward[1]-100]
                else:
                    shaped_reward = [-1, -1]
            if self.train_env.env_core.current_team == self.agent_idx:
                shaped_reward[self.agent_idx] = - euclidean_distance([300, 500], agent_pos)
            return shaped_reward


        def evaluate(self, **kwargs):
            return {}
        def save_video_demo(self, **kwargs):
            return