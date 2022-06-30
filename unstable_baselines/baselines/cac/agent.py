import torch
import torch.nn.functional as F
from operator import itemgetter
from torch import nn

from unstable_baselines.baselines.cac.utils import th_grads_flatten, AdaptiveMultiObjective
from unstable_baselines.common.agents import BaseAgent
# from unstable_baselines.common.networks import  MLPNetwork, PolicyNetwork, get_optimizer
from unstable_baselines.common.networks import MLPNetwork, PolicyNetworkFactory, get_optimizer
import numpy as np
from unstable_baselines.common import util, functional


class CACAgent(torch.nn.Module, BaseAgent):
    def __init__(self, observation_space, action_space,
                 target_smoothing_tau,
                 alpha,
                 reward_scale,
                 kl_forward,
                 **kwargs):
        super(CACAgent, self).__init__()
        self._kl_forward = kl_forward
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]

        # initilize networks
        self.q1_network = MLPNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.q2_network = MLPNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.target_q1_network = MLPNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.target_q2_network = MLPNetwork(obs_dim + action_dim, 1, **kwargs['q_network'])
        self.policy_network = PolicyNetworkFactory.get(obs_dim, action_space, **kwargs["policy_network"])
        # initialize forward network
        self._forward_net = MLPNetwork(obs_dim + action_dim, obs_dim, **kwargs['forward_network'])

        # sync network parameters
        functional.soft_update_network(self.q1_network, self.target_q1_network, 1.0)
        functional.soft_update_network(self.q2_network, self.target_q2_network, 1.0)

        # pass to util.device
        self.q1_network = self.q1_network.to(util.device)
        self.q2_network = self.q2_network.to(util.device)
        self.target_q1_network = self.target_q1_network.to(util.device)
        self.target_q2_network = self.target_q2_network.to(util.device)
        self.policy_network = self.policy_network.to(util.device)
        self._forward_net = self._forward_net.to(util.device)

        # register networks
        self.networks = {
            'q1_network': self.q1_network,
            'q2_network': self.q2_network,
            'target_q1_network': self.target_q1_network,
            'target_q2_network': self.target_q2_network,
            'policy_network': self.policy_network,
            '_forward_net': self._forward_net
        }

        # initialize optimizer
        self.q1_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q1_network,
                                          kwargs['q_network']['learning_rate'])
        self.q2_optimizer = get_optimizer(kwargs['q_network']['optimizer_class'], self.q2_network,
                                          kwargs['q_network']['learning_rate'])
        self.policy_optimizer = get_optimizer(kwargs['policy_network']['optimizer_class'], self.policy_network,
                                              kwargs['policy_network']['learning_rate'])
        self._forward_optimizer = get_optimizer(kwargs['forward_network']['optimizer_class'], self._forward_net,
                                                kwargs['forward_network']['learning_rate'])

        # entropy
        self.automatic_entropy_tuning = kwargs['entropy']['automatic_tuning']
        self.alpha = alpha
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, device=util.device)
            self.log_alpha = nn.Parameter(self.log_alpha, requires_grad=True)
            # sel5f.log_alpha = torch.FloatTensor(math.log(alpha), requires_grad=True, device=util.device)
            self.alpha = self.log_alpha.detach().exp()
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=kwargs['entropy']['learning_rate'])

        self._beta = torch.tensor(0.).to(util.device)
        self._adap = AdaptiveMultiObjective(obj_nums=2)  # main, alpha, and beta.
        self._adap.set_alpha_lr_for_cac(kwargs['policy_network']['learning_rate'])

        # hyper-parameters
        self.gamma = kwargs['gamma']
        self.target_smoothing_tau = target_smoothing_tau
        self.reward_scale = reward_scale

    def update(self, data_batch):
        obs_batch = data_batch['obs']
        action_batch = data_batch['action']
        next_obs_batch = data_batch['next_obs']
        reward_batch = data_batch['reward']
        done_batch = data_batch['done']

        # update forward model
        predict_next_obs = self._forward_net(torch.cat([obs_batch, action_batch], dim=1))
        forward_loss = F.mse_loss(predict_next_obs, next_obs_batch)
        self._forward_optimizer.zero_grad()
        forward_loss.backward()
        self._forward_optimizer.step()

        reward_batch = reward_batch * self.reward_scale
        curr_state_q1_value = self.q1_network(torch.cat([obs_batch, action_batch], dim=1))
        curr_state_q2_value = self.q2_network(torch.cat([obs_batch, action_batch], dim=1))
        with torch.no_grad():
            next_state_action, next_state_log_pi = \
                itemgetter("action_scaled", "log_prob")(self.policy_network.sample(next_obs_batch))

            state_t2 = self._forward_net(torch.cat([next_obs_batch, next_state_action], dim=1)).detach()
            state_t2_action, state_t2_log_pi = \
                itemgetter("action_scaled", "log_prob")(self.policy_network.sample(state_t2))

            next_state_q1_value = self.target_q1_network(torch.cat([next_obs_batch, next_state_action], dim=1))
            next_state_q2_value = self.target_q2_network(torch.cat([next_obs_batch, next_state_action], dim=1))
            next_state_min_q = torch.min(next_state_q1_value, next_state_q2_value)
            if self._kl_forward:
                target_q = next_state_min_q - self.alpha * next_state_log_pi - self._beta * (
                        next_state_log_pi - state_t2_log_pi)
            else:
                target_q = next_state_min_q - self.alpha * next_state_log_pi - self._beta * (
                        state_t2_log_pi - next_state_log_pi)
            target_q = reward_batch + self.gamma * (1. - done_batch) * target_q

        # compute q loss and backward
        q1_loss = F.mse_loss(curr_state_q1_value, target_q)
        q2_loss = F.mse_loss(curr_state_q2_value, target_q)

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        ########## actor

        new_curr_state_action, new_curr_state_log_pi = \
            itemgetter("action_scaled", "log_prob")(self.policy_network.sample(obs_batch))
        state_t1 = self._forward_net(torch.cat([obs_batch, new_curr_state_action], dim=1)).detach()

        state_t1_action, state_t1_log_pi = \
            itemgetter("action_scaled", "log_prob")(self.policy_network.sample(state_t1))

        new_curr_state_q1_value = self.q1_network(torch.cat([obs_batch, new_curr_state_action], dim=1))
        new_curr_state_q2_value = self.q2_network(torch.cat([obs_batch, new_curr_state_action], dim=1))
        new_min_curr_state_q_value = torch.min(new_curr_state_q1_value, new_curr_state_q2_value)

        # compute policy and ent loss
        policy_loss = ((self.alpha * new_curr_state_log_pi) - new_min_curr_state_q_value).mean()

        # kl loss
        if self._kl_forward:
            policy_kl_loss = new_curr_state_log_pi - state_t1_log_pi
        else:
            policy_kl_loss = state_t1_log_pi - new_curr_state_log_pi
        policy_kl_loss = policy_kl_loss.mean()

        main_policy_grads_flat = th_grads_flatten(policy_loss, self.policy_network, retain_graph=True).detach()
        kl_policy_grads_flat = th_grads_flatten(policy_kl_loss, self.policy_network, retain_graph=True).detach()

        _, _beta = self._adap.calculate_grad_weights(
            grads=[main_policy_grads_flat, kl_policy_grads_flat],
            last_weights=[1., self._beta],
            mode='cosine_grad')
        self._beta.copy_(_beta)

        policy_tot_loss = policy_loss + self._beta * policy_kl_loss

        self.policy_optimizer.zero_grad()
        policy_tot_loss.backward()
        self.policy_optimizer.step()

        # alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (new_curr_state_log_pi + self.target_entropy).detach()).mean()
            alpha_loss_value = alpha_loss.detach().cpu().item()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()
        else:
            alpha_loss = 0.
            alpha_loss_value = 0.

        self.update_target_network()

        return {
            "loss/q1": q1_loss.item(),
            "loss/q2": q2_loss.item(),
            "loss/policy": policy_loss.item(),
            "loss/policy_kl_loss": policy_kl_loss.item(),
            "loss/policy_tot_loss": policy_tot_loss.item(),
            "loss/forward": forward_loss.item(),
            "loss/entropy": alpha_loss_value,
            "misc/entropy_alpha": self.alpha.item(),
            "misc/beta": self._beta.item(),
        }

    def update_target_network(self):
        functional.soft_update_network(self.q1_network, self.target_q1_network, self.target_smoothing_tau)
        functional.soft_update_network(self.q2_network, self.target_q2_network, self.target_smoothing_tau)

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        if len(obs.shape) == 1:
            ret_single = True
            obs = [obs]
        if type(obs) != torch.tensor:
            obs = torch.FloatTensor(np.array(obs)).to(util.device)
        action, log_prob = itemgetter("action_scaled", "log_prob")(
            self.policy_network.sample(obs, deterministic=deterministic))
        if ret_single:
            action = action[0]
            log_prob = log_prob[0]
        return {
            'action': action.detach().cpu().numpy(),
            'log_prob': log_prob
        }
