import torch as th

from collections import deque, defaultdict
from copy import deepcopy
from typing import List

import torch.nn.functional as F
import numpy as np


def th_grads_flatten(loss, model, **kwargs):
    grads = th.autograd.grad(loss, model.parameters(), **kwargs)
    return th.cat([grad.reshape(-1) for grad in grads])


def calculate_smooth_statistics(trajs: List[List], interval=20):
    """
    trajs: List of traj
        traj: List of (s, a)
    """
    ret_dict = defaultdict(list)

    for traj in trajs:
        _length = len(traj)

        _actions_list = [_exp[-1] for _exp in traj]

        idxs = np.arange(_length, step=interval)
        idxs = np.append(idxs, _length)
        # index from 0 to _length - 1

        abs_diff = []
        for _i, _j in zip(idxs[:-1], idxs[1:]):
            acts = _actions_list[_i:_j]
            _abs_diff = 0.
            for pre_act, post_act in zip(acts[:-1], acts[1:]):
                _abs_diff += np.sum(np.abs(pre_act - post_act))
            abs_diff.append(_abs_diff / (_j - _i))

        _var = np.var(abs_diff)

        acts = _actions_list[0:_length]
        _abs_diff = 0.
        for pre_act, post_act in zip(acts[:-1], acts[1:]):
            _abs_diff += np.sum(np.abs(pre_act - post_act))
        _tv = _abs_diff / _length

        ret_dict['smooth/smooth_var'].append(_var)
        ret_dict['smooth/smooth_tv'].append(_tv)

    for k in ret_dict.keys():
        ret_dict[k] = np.mean(np.asarray(ret_dict[k]))

    return ret_dict


class AdaptiveMultiObjective(object):
    mode = [
        'cosine_similarity_normalization',
        'cosine_grad',
        'OL-AUX',  # 《Adaptive Auxiliary Task Weighting for Reinforcement Learning》
        'original_weighted',  # http://arxiv.org/abs/1812.02224
        'original_unweighted',  # http://arxiv.org/abs/1812.02224
        'fixed'
    ]

    def __init__(self, obj_nums=1):
        """

        """
        self._obs_nums = obj_nums

        # for cac
        self._grads_buffer_maxlen = 1
        self._grads_buffer = deque(maxlen=self._grads_buffer_maxlen)

        self._ol_aux_N = 5
        self._ol_aux_lr = 5. * self._ol_aux_N
        self._ol_aux_list = deque(maxlen=self._ol_aux_N)

    def _check_length(self, obj):
        """

        """
        assert len(obj) == self._obs_nums, "assert len(obj) == self._obs_nums"

    def calculate_grad_weights(self, grads: List, last_weights: List = None, mode: str = None):
        """

        """
        mode = mode or AdaptiveMultiObjective.mode[0]
        assert mode in AdaptiveMultiObjective.mode, "assert mode in AdaptiveMultiObjective.mode"

        self._check_length(grads)
        last_weights = last_weights or [1.] * self._obs_nums
        self._check_length(last_weights)

        return self._adap_for_cac2(grads, last_weights, mode)

    def _adap_for_cac(self, grads: List, last_weights: List = None, mode: str = None):
        """

        """
        main_actor_grads_flat, alpha_grads_flat, beta_grads_flat = grads
        self._grads_buffer.append((deepcopy(main_actor_grads_flat),
                                   deepcopy(alpha_grads_flat),
                                   deepcopy(beta_grads_flat)))
        main_actor_grads_flat = sum([grad[0] for grad in self._grads_buffer]) / self._grads_buffer_maxlen
        alpha_grads_flat = sum([grad[1] for grad in self._grads_buffer]) / self._grads_buffer_maxlen
        beta_grads_flat = sum([grad[2] for grad in self._grads_buffer]) / self._grads_buffer_maxlen

        _, alpha, beta = last_weights

        alpha_coff = F.cosine_similarity(main_actor_grads_flat, alpha_grads_flat, dim=0)  # [-1, 1]
        beta_coff = F.cosine_similarity(main_actor_grads_flat, beta_grads_flat, dim=0)  # [-1, 1]

        if mode == 'cosine_similarity_normalization':
            idx_factor = 2
            # [-1, 1] => [0, 1]
            alpha_coff = (alpha_coff + 1) / 2  # [0, 1]
            beta_coff = (beta_coff + 1) / 2  # [0, 1]
            self._grads_buffer.append((alpha_coff ** idx_factor,
                                       beta_coff ** idx_factor))

            if len(self._grads_buffer) == self._grads_buffer_maxlen:
                alpha = sum([grad[0] for grad in self._grads_buffer]) / self._grads_buffer_maxlen
                beta = sum([grad[1] for grad in self._grads_buffer]) / self._grads_buffer_maxlen

        elif mode == 'OL-AUX':  # 《Adaptive Auxiliary Task Weighting for Reinforcement Learning》
            _alpha_grad = -self._actor_lr * (main_actor_grads_flat * alpha_grads_flat).sum()
            _beta_grad = -self._actor_lr * (main_actor_grads_flat * beta_grads_flat).sum()
            self._ol_aux_list.append([_alpha_grad, _beta_grad])
            if len(self._ol_aux_list) == self._ol_aux_N:
                _ag, _bg = 0., 0.
                for (_alpha_grad, _beta_grad) in self._ol_aux_list:
                    _ag += _alpha_grad
                    _bg += _beta_grad
                alpha -= self._ol_aux_lr * _ag
                beta -= self._ol_aux_lr * _bg
                self._ol_aux_list.clear()

        elif mode == 'original_weighted':  # http://arxiv.org/abs/1812.02224
            alpha = alpha_coff.clip(min=0.)
            beta = beta_coff.clip(min=0.)

        elif mode == 'original_unweighted':  # http://arxiv.org/abs/1812.02224
            alpha = (alpha_coff.sign() + 1) / 2  # {0, 0.5, 1}
            beta = (beta_coff.sign() + 1) / 2  # {0, 0.5, 1}

        elif mode == 'fixed':
            alpha = 0.
            beta = 0.

        return [1., alpha, beta]

    def _adap_for_cac2(self, grads: List, last_weights: List = None, mode: str = None):
        """

        """
        main_actor_grads_flat, beta_grads_flat = grads
        self._grads_buffer.append((deepcopy(main_actor_grads_flat),
                                   deepcopy(beta_grads_flat)))
        main_actor_grads_flat = sum([grad[0] for grad in self._grads_buffer]) / self._grads_buffer_maxlen
        beta_grads_flat = sum([grad[1] for grad in self._grads_buffer]) / self._grads_buffer_maxlen

        _, beta = last_weights

        beta_coff = F.cosine_similarity(main_actor_grads_flat, beta_grads_flat, dim=0)  # [-1, 1]

        if mode == 'cosine_similarity_normalization':
            idx_factor = 1
            # [-1, 1] => [0, 1]
            beta_coff = (beta_coff + 1) / 2  # [0, 1]
            self._grads_buffer.append(beta_coff ** idx_factor)

            if len(self._grads_buffer) == self._grads_buffer_maxlen:
                beta = sum(self._grads_buffer) / self._grads_buffer_maxlen

        elif mode == 'cosine_grad':
            # _beta_shift = beta_coff - 0.5  # [-1, 1] => [-1.5, 0.5]
            # if _beta_shift > 0:
            #     _beta_shift *= 2
            # else:
            #     _beta_shift = _beta_shift * 2 / 3
            # beta += 0.1 * (_beta_shift ** 3)  # [-1, 1] => [-0.1, 0.1]
            beta += 0.1 * (beta_coff ** 3)  # [-1, 1] => [-0.1, 0.1]
            beta = beta.clamp(0., 1.)

        elif mode == 'OL-AUX':  # 《Adaptive Auxiliary Task Weighting for Reinforcement Learning》
            _beta_grad = -self._actor_lr * (main_actor_grads_flat * beta_grads_flat).sum()
            self._ol_aux_list.append(_beta_grad)
            if len(self._ol_aux_list) == self._ol_aux_N:
                _bg = 0.
                for _beta_grad in self._ol_aux_list:
                    _bg += _beta_grad
                beta -= max(min(self._ol_aux_lr * _bg, 0.1), -0.1)
                # beta = max(min(beta, 1.0), 0.0)
                self._ol_aux_list.clear()

        elif mode == 'original_weighted':  # http://arxiv.org/abs/1812.02224
            beta = beta_coff.clip(min=0.)

        elif mode == 'original_unweighted':  # http://arxiv.org/abs/1812.02224
            beta = (beta_coff.sign() + 1) / 2  # {0, 0.5, 1}

        elif mode == 'fixed':
            pass

        return [1., beta]

    def set_alpha_lr_for_cac(self, actor_lr):
        self._actor_lr = actor_lr
