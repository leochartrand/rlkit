import os
import os.path as osp
import uuid

from rlkit.envs.vae_wrappers import VAEWrappedEnv, ConditionalVAEWrappedEnv

filename = str(uuid.uuid4())

import skvideo.io
import numpy as np
import time

import scipy.misc

from multiworld.core.image_env import ImageEnv
from rlkit.core import logger
from rlkit.visualization.image import add_border, make_image_fit_into_hwc_format, combine_images_into_grid
import pickle


def save_paths(algo, epoch):
    expl_paths = algo.expl_data_collector.get_epoch_paths()
    filename = osp.join(logger.get_snapshot_dir(),
                        'video_{epoch}_vae.p'.format(epoch=epoch))
    pickle.dump(expl_paths, open(filename, "wb"))
    print("saved", filename)
    eval_paths = algo.eval_data_collector.get_epoch_paths()
    filename = osp.join(logger.get_snapshot_dir(),
                        'video_{epoch}_env.p'.format(epoch=epoch))
    pickle.dump(eval_paths, open(filename, "wb"))
    print("saved", filename)


class VideoSaveFunction:
    def __init__(self, env, variant, expl_path_collector=None,
                 eval_path_collector=None):
        self.env = env
        self.logdir = logger.get_snapshot_dir()
        self.dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        if 'imsize' not in self.dump_video_kwargs:
            self.dump_video_kwargs['imsize'] = env.imsize
        self.dump_video_kwargs.setdefault("rows", 2)
        # self.dump_video_kwargs.setdefault("columns", 5)
        self.dump_video_kwargs.setdefault("columns", 1)
        self.dump_video_kwargs.setdefault("unnormalize", True)
        self.save_period = self.dump_video_kwargs.pop('save_video_period', 50)
        self.exploration_goal_image_key = self.dump_video_kwargs.pop(
            "exploration_goal_image_key", "decoded_goal_image")
        self.evaluation_goal_image_key = self.dump_video_kwargs.pop(
            "evaluation_goal_image_key", "image_desired_goal")
        self.path_length = variant.get('algo_kwargs', {}).get('max_path_length', 200)
        self.expl_path_collector = expl_path_collector
        self.eval_path_collector = eval_path_collector
        self.variant = variant

    def __call__(self, algo, epoch):
        if self.expl_path_collector:
            expl_paths = self.expl_path_collector.collect_new_paths(
                max_path_length=self.path_length,
                num_steps=self.path_length * 5,
                discard_incomplete_paths=False
            )
        else:
            expl_paths = algo.expl_data_collector.get_epoch_paths()
        if epoch % self.save_period == 0 or epoch >= algo.num_epochs - 1:
            filename = osp.join(self.logdir,
                                'video_{epoch}_vae.mp4'.format(epoch=epoch))
            dump_paths(self.env,
                       filename,
                       expl_paths,
                       [self.exploration_goal_image_key, "image_observation", ],
                       **self.dump_video_kwargs,
                       )

        if self.eval_path_collector:
            eval_paths = self.eval_path_collector.collect_new_paths(
                max_path_length=self.path_length,
                num_steps=self.path_length * 5,
                discard_incomplete_paths=False
            )
        else:
            eval_paths = algo.eval_data_collector.get_epoch_paths()
        if epoch % self.save_period == 0 or epoch >= algo.num_epochs - 1:
            filename = osp.join(self.logdir,
                                'video_{epoch}_env.mp4'.format(epoch=epoch))
            dump_paths(self.env,
                       filename,
                       eval_paths,
                       [self.evaluation_goal_image_key, "image_observation", ],
                       **self.dump_video_kwargs,
                       )


class RIGVideoSaveFunction:
    def __init__(self,
        model,
        data_collector,
        tag,
        save_video_period,
        goal_image_key=None,
        decode_goal_image_key=None,
        reconstruction_key=None,
        **kwargs
    ):
        self.model = model
        self.data_collector = data_collector
        self.tag = tag
        self.goal_image_key = goal_image_key
        self.decode_goal_image_key = decode_goal_image_key
        self.reconstruction_key = reconstruction_key
        self.dump_video_kwargs = kwargs
        self.save_video_period = save_video_period
        self.keys = []
        if goal_image_key:
            self.keys.append(goal_image_key)
        if decode_goal_image_key:
            self.keys.append(decode_goal_image_key)
        self.keys.append("image_observation")
        if reconstruction_key:
            self.keys.append(reconstruction_key)
        self.logdir = logger.get_snapshot_dir()

    def __call__(self, algo, epoch):
        paths = self.data_collector.get_epoch_paths()
        if epoch % self.save_video_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(self.logdir,
                'video_{epoch}_{tag}.mp4'.format(epoch=epoch, tag=self.tag))
            if self.decode_goal_image_key:
                for i in range(len(paths)):
                    self.add_decoded_goal_to_path(paths[i])
            if self.reconstruction_key:
                for i in range(len(paths)):
                    self.add_reconstruction_to_path(paths[i])
            dump_paths(None,
                filename,
                paths,
                self.keys,
                **self.dump_video_kwargs,
            )

    def add_decoded_goal_to_path(self, path):
        latent = path['full_observations'][0]['latent_desired_goal']
        decoded_img = self.model.decode_one_np(latent)
        for i_in_path, d in enumerate(path['full_observations']):
            d[self.decode_goal_image_key] = decoded_img

    def add_reconstruction_to_path(self, path):
        for i_in_path, d in enumerate(path['full_observations']):
            latent = d['latent_observation']
            decoded_img = self.model.decode_one_np(latent)
            d[self.reconstruction_key] = decoded_img


def dump_video(
        env,
        policy,
        filename,
        rollout_function,
        rows=3,
        columns=6,
        do_timer=True,
        horizon=100,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=84,
        get_extra_imgs=None,
        grayscale=False,
        keys_to_show=None,
        num_columns_per_rollout=1,
        obs_dict_key='full_observations',
        **combine_img_kwargs
):
    """

    :param env:
    :param policy:
    :param filename:
    :param rollout_function:
    :param rows:
    :param columns:
    :param pad_length:
    :param subpad_color:
    :param do_timer:
    :param horizon:
    :param dirname_to_save_images:
    :param subdirname:
    :param imsize: TODO: automatically set if not provided
    :param get_extra_imgs: A function with type

        def get_extra_imgs(
            path: List[dict],
            index_in_path: int,
            env,
        ) -> List[np.ndarray]:
    :param grayscale:
    :return:
    """
    if get_extra_imgs is None:
        get_extra_imgs = get_generic_env_imgs
    num_channels = 1 if grayscale else 3
    keys_to_show = keys_to_show or ['image_desired_goal', 'image_observation']
    frames = []
    N = rows * columns
    for i in range(N):
        start = time.time()
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
        )

        l = []
        for i_in_path, d in enumerate(path[obs_dict_key]):
            imgs_to_stack = [d[k] for k in keys_to_show]
            imgs_to_stack += get_extra_imgs(path, i_in_path, env)
            grid_img = combine_images_into_grid(
                imgs_to_stack,
                max_num_cols=num_columns_per_rollout,
                imwidth=imsize,
                imheight=imsize,
                unnormalize=True,
                **combine_img_kwargs
            )
            l.append(grid_img)
        if len(l) < horizon:
            frozen_img = l[-1] / 2
            l += [frozen_img] * (horizon - len(l))
        frames += l

        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-101:]
            goal_img = np.flip(rollout_frames[0][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/goal.png", goal_img)
            goal_img = np.flip(rollout_frames[1][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/z_goal.png", goal_img)
            for j in range(0, 101, 1):
                img = np.flip(rollout_frames[j][imsize:, :imsize, :], 0)
                scipy.misc.imsave(rollout_dir + "/" + str(j) + ".png", img)
        if do_timer:
            print(i, time.time() - start)

    outputdata = reshape_for_video(frames, N, rows, columns, num_channels)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)


def reshape_for_video(frames, N, rows, columns, num_channels):
    img_height, img_width = frames[0].shape[:2]
    frames = np.array(frames, dtype=np.uint8)
    # TODO: can't we just do path_length = len(frames) / N ?
    path_length = frames.size // (
            N * img_height * img_width * num_channels
    )
    frames = np.array(frames, dtype=np.uint8).reshape(
        (N, path_length, img_height, img_width, num_channels)
    )
    f1 = []
    for k1 in range(columns):
        f2 = []
        for k2 in range(rows):
            k = k1 * rows + k2
            f2.append(frames[k:k + 1, :, :, :, :].reshape(
                (path_length, img_height, img_width, num_channels)
            ))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)
    return outputdata


def get_generic_env_imgs(path, i_in_path, env):
    is_vae_env = isinstance(env, VAEWrappedEnv)
    is_conditional_vae_env = isinstance(env, ConditionalVAEWrappedEnv)
    imgs = []
    if not is_conditional_vae_env or not is_vae_env:
        return imgs
    x_0 = path['full_observations'][0]['image_observation']
    d = path['full_observations'][i_in_path]
    if is_conditional_vae_env:
        imgs.append(
            np.clip(env._reconstruct_img(d['image_observation'], x_0), 0, 1)
        )
    elif is_vae_env:
        imgs.append(
            np.clip(env._reconstruct_img(d['image_observation']), 0, 1)
        )
    return imgs


def dump_paths(
        env,
        filename,
        paths,
        keys,
        rows=3,
        columns=6,
        do_timer=True,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=84,  # TODO: automatically set if not provided
        imwidth=None,
        imheight=None,
        num_imgs=3,  # how many vertical images we stack per rollout
        dump_pickle=False,
        grayscale=False,
        get_extra_imgs=None,
        num_columns_per_rollout=1,
        **combine_img_kwargs
):
    # TODO: merge with `dump_video`
    if get_extra_imgs is None:
        get_extra_imgs = get_generic_env_imgs
    # num_channels = env.vae.input_channels
    num_channels = 1 if grayscale else 3
    frames = []

    imwidth = imwidth or imsize  # 500
    imheight = imheight or imsize  # 300
    num_gaps = num_imgs - 1  # 2

    H = num_imgs * imheight  # imsize
    W = imwidth  # imsize

    if len(paths) < rows * columns:
        columns = min(columns, len(paths))
        rows = max(min(rows, int(len(paths) / columns)), 1)
    else:
        rows = min(rows, int(len(paths) / columns))
    N = rows * columns
    for i in range(N):
        start = time.time()
        path = paths[i]
        l = []
        for i_in_path, d in enumerate(path['full_observations']):
            imgs = [d[k] for k in keys]
            imgs = imgs + get_extra_imgs(path, i_in_path, env)
            imgs = imgs[:num_imgs]
            l.append(
                combine_images_into_grid(
                    imgs,
                    imwidth,
                    imheight,
                    max_num_cols=num_columns_per_rollout,
                    **combine_img_kwargs
                )
            )
        frames += l

        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-101:]
            goal_img = np.flip(rollout_frames[0][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/goal.png", goal_img)
            goal_img = np.flip(rollout_frames[1][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/z_goal.png", goal_img)
            for j in range(0, 101, 1):
                img = np.flip(rollout_frames[j][imsize:, :imsize, :], 0)
                scipy.misc.imsave(rollout_dir + "/" + str(j) + ".png", img)
        if do_timer:
            print(i, time.time() - start)

    outputdata = reshape_for_video(frames, N, rows, columns, num_channels)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)

    if dump_pickle:
        pickle_filename = filename[:-4] + ".p"
        pickle.dump(paths, open(pickle_filename, "wb"))


def get_save_video_function(
        rollout_function,
        env,
        policy,
        save_video_period=10,
        imsize=48,
        tag="",
        video_image_env_kwargs=None,
        **dump_video_kwargs
):
    logdir = logger.get_snapshot_dir()

    if not isinstance(env, ImageEnv) and not isinstance(env, VAEWrappedEnv):
        if video_image_env_kwargs is None:
            video_image_env_kwargs = {}
        image_env = ImageEnv(env, imsize, transpose=True, normalize=True,
                             **video_image_env_kwargs)
    else:
        image_env = env
        assert image_env.imsize == imsize, "Imsize must match env imsize"

    def save_video(algo, epoch):
        if epoch % save_video_period == 0 or epoch >= algo.num_epochs - 1:
            filename = osp.join(
                logdir,
                'video_{}_{epoch}_env.mp4'.format(tag, epoch=epoch),
            )
            dump_video(image_env, policy, filename, rollout_function,
                       imsize=imsize, **dump_video_kwargs)
    return save_video
