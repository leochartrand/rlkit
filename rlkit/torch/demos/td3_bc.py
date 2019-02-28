from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm

from rlkit.util.io import load_local_or_remote_file

import random
from rlkit.torch.core import PyTorchModule, torch_ify
from rlkit.data_management.path_builder import PathBuilder
import time

class TD3BC(TorchRLAlgorithm):
    """
    Twin Delayed Deep Deterministic policy gradients
    """

    def __init__(
            self,
            env,
            qf1,
            qf2,
            policy,
            exploration_policy,
            demo_path,
            demo_train_buffer,
            demo_test_buffer,
            demo_num_paths=1,
            apply_her_to_demos=False,
            add_demo_latents=False,
            demo_train_split=0.9,
            add_demos_to_replay_buffer=True,
            bc_weight=0.0,
            rl_weight=1.0,
            weight_decay=0,
            eval_policy=None,
            use_conditional_vae=True,

            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,

            policy_learning_rate=1e-3,
            qf_learning_rate=1e-3,
            policy_and_target_update_period=2,
            tau=0.005,
            qf_criterion=None,
            optimizer_class=optim.Adam,

            **kwargs
    ):

        super().__init__(
            env,
            exploration_policy,
            eval_policy=eval_policy or policy,
            **kwargs
        )
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        self.qf1 = qf1
        self.qf2 = qf2
        self.policy = policy

        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip

        self.policy_and_target_update_period = policy_and_target_update_period
        self.tau = tau
        self.qf_criterion = qf_criterion

        self.target_policy = policy.copy()
        self.target_qf1 = self.qf1.copy()
        self.target_qf2 = self.qf2.copy()
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_learning_rate,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_learning_rate,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_learning_rate,
            weight_decay=weight_decay,
        )
        self.bc_weight = bc_weight
        self.rl_weight = rl_weight

        self.demo_num_paths = demo_num_paths
        self.add_demos_to_replay_buffer = add_demos_to_replay_buffer
        self.demo_train_split = demo_train_split
        self.demo_train_buffer = demo_train_buffer
        self.demo_test_buffer = demo_test_buffer
        self.add_demo_latents = add_demo_latents
        self.apply_her_to_demos = apply_her_to_demos

        self.demo_path = demo_path
        self.use_conditional_vae = use_conditional_vae
        self.load_demos(self.demo_path)


    def _update_obs_with_latent(self, obs):
        latent_obs = self.env._encode_one(obs["image_observation"])
        latent_goal = self.env._encode_one(obs["image_desired_goal"])
        obs['latent_observation'] = latent_obs
        obs['latent_achieved_goal'] = latent_obs
        obs['latent_desired_goal'] = latent_goal
        obs['observation'] = latent_obs
        obs['achieved_goal'] = latent_obs
        obs['desired_goal'] = latent_goal
        return obs


    def _update_obs_with_latent_batch(self, obs, latent_goal):
        latent_obs = self.env._encode(np.array([ob["image_observation"] for ob in obs]) / 255.0)
        # latent_goals = self.env._encode(np.array([ob["image_desired_goal"] for ob in obs])/ 255.0)
        for ob, latent_ob in zip(obs, latent_obs):
            ob['latent_observation'] = latent_ob
            ob['latent_achieved_goal'] = latent_ob
            ob['latent_desired_goal'] = latent_goal
            ob['observation'] = latent_ob
            ob['achieved_goal'] = latent_ob
            ob['desired_goal'] = latent_goal
        return obs

    # def load_path(self, path, replay_buffer):
    #     # final_achieved_goal = path["observations"][-1]["state_achieved_goal"].copy()
    #     path_builder = PathBuilder()
    #     import time
    #     for (
    #         ob,
    #         action,
    #         reward,
    #         next_ob,
    #         terminal,
    #         agent_info,
    #         env_info,
    #     ) in zip(
    #         path["observations"],
    #         path["actions"],
    #         path["rewards"],
    #         path["next_observations"],
    #         path["terminals"],
    #         path["agent_infos"],
    #         path["env_infos"],
    #     ):
    #         # goal = path["goal"]["state_desired_goal"][0, :]
    #         # import pdb; pdb.set_trace()
    #         # print(goal.shape, ob["state_observation"])
    #         t0 = time.time()
    #         # state_observation = np.concatenate((ob["state_observation"], goal))
    #         action = action[:3]
    #         if "image_desired_goal" not in ob.keys():
    #             ob["image_desired_goal"] = path['observations'][-1]['image_observation']
    #
    #         if "image_desired_goal" not in next_ob.keys():
    #             next_ob["image_desired_goal"] = path['next_observations'][-1]['image_observation']
    #
    #
    #         if self.add_demo_latents:
    #             ob["image_observation"] = ob["image_observation"] / 255.0
    #             ob["image_desired_goal"] = ob["image_desired_goal"] / 255.0
    #             next_ob["image_observation"] = next_ob["image_observation"] / 255.0
    #             next_ob["image_desired_goal"] = next_ob["image_desired_goal"] / 255.0
    #             t0 = time.time()
    #             self._update_obs_with_latent(ob)
    #             t1 = time.time()
    #             self._update_obs_with_latent(next_ob)
    #             t2 = time.time()
    #
    #             reward = self.env.compute_reward(
    #                 action,
    #                 {'latent_achieved_goal': next_ob['latent_achieved_goal'],
    #                  'latent_desired_goal': next_ob['latent_desired_goal']}
    #             )
    #         if self.apply_her_to_demos:
    #             ob["state_desired_goal"] = final_achieved_goal
    #             next_ob["state_desired_goal"] = final_achieved_goal
    #             reward = self.env.compute_reward(
    #                 action,
    #                 next_ob,
    #             )
    #         reward = np.array([reward])
    #         terminal = np.array([terminal])
    #         path_builder.add_all(
    #             observations=ob,
    #             actions=action,
    #             rewards=reward,
    #             next_observations=next_ob,
    #             terminals=terminal,
    #             agent_infos=agent_info,
    #             env_infos=env_info,
    #         )
    #
    #     path = path_builder.get_all_stacked()
    #
    #     replay_buffer.add_path(path)




# GOAL CONDITIONED, COMMENTED FOR NOW
    # def load_path(self, path, replay_buffer, batch_size=16):
    #         # final_achieved_goal = path["observations"][-1]["state_achieved_goal"].copy()
    #     path_builder = PathBuilder()
    #     import time

    #     N = path['observations']

    #     t0 = time.time()

    #     # for ob, next_ob in zip(path['observations'], path['next_observations']):

    #     #     ob["image_desired_goal"] = path['observations'][-1]['image_observation']

    #     #     next_ob["image_desired_goal"] = path['next_observations'][-1]['image_observation']

    #         # if self.add_demo_latents:
    #         #     ob["image_observation"] = ob["image_observation"] / 255.0
    #         #     ob["image_desired_goal"] = ob["image_desired_goal"] / 255.0
    #         #     next_ob["image_observation"] = next_ob["image_observation"] / 255.0
    #         #     next_ob["image_desired_goal"] = next_ob["image_desired_goal"] / 255.0
    #     t1 = time.time()
    #     # print('transformations', t1 - t0)

    #     t2 = time.time()
    #     if self.add_demo_latents:
    #         image_goal = path['next_observations'][-1]['image_observation']
    #         latent_goal = self.env._encode_one(image_goal / 255.0)
    #         self._update_obs_with_latent_batch(path['observations'], latent_goal)
    #         self._update_obs_with_latent_batch(path['next_observations'], latent_goal)
    #     t3 = time.time()
    #     # print('Time to update', t3 - t2)
    #     for (
    #             ob,
    #             action,
    #             reward,
    #             next_ob,
    #             terminal,
    #             agent_info,
    #             env_info,
    #     ) in zip(
    #         path["observations"],
    #         path["actions"],
    #         path["rewards"],
    #         path["next_observations"],
    #         path["terminals"],
    #         path["agent_infos"],
    #         path["env_infos"],
    #     ):

    #         t0 = time.time()
    #         # state_observation = np.concatenate((ob["state_observation"], goal))
    #         action = action[:3]
    #         if self.add_demo_latents:
    #             reward = self.env.compute_reward(
    #                 action,
    #                 {'latent_achieved_goal': next_ob['latent_achieved_goal'],
    #                  'latent_desired_goal': next_ob['latent_desired_goal']}
    #             )
    #         if self.apply_her_to_demos:
    #             ob["state_desired_goal"] = final_achieved_goal
    #             next_ob["state_desired_goal"] = final_achieved_goal
    #             reward = self.env.compute_reward(
    #                 action,
    #                 next_ob,
    #             )
    #         reward = np.array([reward])
    #         terminal = np.array([terminal])
    #         path_builder.add_all(
    #             observations=ob,
    #             actions=action,
    #             rewards=reward,
    #             next_observations=next_ob,
    #             terminals=terminal,
    #             agent_infos=agent_info,
    #             env_infos=env_info,
    #         )

    #     path = path_builder.get_all_stacked()

    #     replay_buffer.add_path(path)

    def load_path(self, path, replay_buffer, batch_size=16):
            # final_achieved_goal = path["observations"][-1]["state_achieved_goal"].copy()
        path_builder = PathBuilder()
        import time

        N = path['observations']

        # import pdb; pdb.set_trace()
        for i in range(len(path["actions"])):
            ob = path["observations"][i]
            action = path["actions"][i]
            reward = path["rewards"][i]
            next_ob = path["observations"][i+1]
            terminal = i == (len(path["actions"]) - 1)
            agent_info = {}
            env_info = {}

            # Hardcoding some transformations
            ob = np.zeros((11))
            next_ob = np.zeros((11))
            action = action[:3]

            reward = np.array([reward])
            terminal = np.array([terminal])
            path_builder.add_all(
                observations=ob,
                actions=action,
                rewards=reward,
                next_observations=next_ob,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )

        path = path_builder.get_all_stacked()

        replay_buffer.add_path(path)

    def load_demos(self, demo_path):
        import time

        t0 = time.time()
        data = load_local_or_remote_file(demo_path)
        random.shuffle(data)
        N = int(len(data) * self.demo_train_split)
        print("using", N, "paths for training")
        self.env.conditional_vae = False
        for path in data[:N]:
            print("loading one path")
            self.load_path(path, self.demo_train_buffer)

        if self.add_demos_to_replay_buffer:
            for path in data[:N]:
                self.load_path(path, self.replay_buffer)

        for path in data[N:]:
            self.load_path(path, self.demo_test_buffer)
        print('Time', time.time() - t0)

        if self.use_conditional_vae:
            self.env.conditional_vae = True

        print('loaded demos')

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        self._train_given_data(
            rewards,
            terminals,
            obs,
            actions,
            next_obs,
        )

    def get_batch_from_buffer(self, replay_buffer):
        batch = replay_buffer.random_batch(self.batch_size)
        for key in batch:
            batch[key] = torch_ify(batch[key])
        # batch = torch_ify(batch)
        # obs = batch['observations']
        # next_obs = batch['next_observations']
        # goals = batch['resampled_goals']
        # batch['observations'] = torch.cat((
        #     obs,
        #     goals
        # ), dim=1)
        # batch['next_observations'] = torch.cat((
        #     next_obs,
        #     goals
        # ), dim=1)
        return batch

    def _train_given_data(
        self,
        rewards,
        terminals,
        obs,
        actions,
        next_obs,
        logger_prefix="",
    ):
        """
        Critic operations.
        """

        next_actions = self.target_policy(next_obs)
        noise = ptu.randn(next_actions.shape) * self.target_policy_noise
        noise = torch.clamp(
            noise,
            -self.target_policy_noise_clip,
            self.target_policy_noise_clip
        )
        noisy_next_actions = next_actions + noise

        target_q1_values = self.target_qf1(next_obs, noisy_next_actions)
        target_q2_values = self.target_qf2(next_obs, noisy_next_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        q1_pred = self.qf1(obs, actions)
        bellman_errors_1 = (q1_pred - q_target) ** 2
        qf1_loss = bellman_errors_1.mean()

        q2_pred = self.qf2(obs, actions)
        bellman_errors_2 = (q2_pred - q_target) ** 2
        qf2_loss = bellman_errors_2.mean()

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        policy_actions = policy_loss = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            policy_actions = self.policy(obs)
            q_output = self.qf1(obs, policy_actions)

            train_batch = self.get_batch_from_buffer(self.demo_train_buffer)
            train_o = train_batch["observations"]
            train_u = train_batch["actions"]
            train_pred_u = self.policy(train_o)
            train_error = (train_pred_u - train_u) ** 2
            train_bc_loss = train_error.mean()

            policy_loss = - self.rl_weight * q_output.mean() + self.bc_weight * train_bc_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            if policy_loss is None:
                policy_actions = self.policy(obs)
                q_output = self.qf1(obs, policy_actions)
                policy_loss = - q_output.mean()

            self.eval_statistics[logger_prefix + 'QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics[logger_prefix + 'QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics[logger_prefix + 'Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics[logger_prefix + 'BC Loss'] = np.mean(ptu.get_numpy(
                train_bc_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Bellman Errors 1',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Bellman Errors 2',
                ptu.get_numpy(bellman_errors_2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                logger_prefix + 'Policy Action',
                ptu.get_numpy(policy_actions),
            ))

            test_batch = self.get_batch_from_buffer(self.demo_test_buffer)
            test_o = test_batch["observations"]
            test_u = test_batch["actions"]
            test_pred_u = self.policy(test_o)
            test_error = (test_pred_u - test_u) ** 2
            test_bc_loss = test_error.mean()
            self.eval_statistics[logger_prefix + 'Test BC Loss'] = np.mean(ptu.get_numpy(
                test_bc_loss
            ))

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        self.update_epoch_snapshot(snapshot)
        return snapshot

    def update_epoch_snapshot(self, snapshot):
        snapshot.update(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.eval_policy,
            trained_policy=self.policy,
            target_policy=self.target_policy,
            exploration_policy=self.exploration_policy,
        )

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_policy,
            self.target_qf1,
            self.target_qf2,
        ]
