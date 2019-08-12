import numpy as np
import pdb
import h5py
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from visual_mpc.agent.utils.utils import AttrDict


def pad_traj_timesteps(traj, max_num_actions):
    """
    pad images and actions with zeros
    :param traj:
    :param max_num_actions:
    :return:
    """

    im_shape = traj.images.shape
    ac_shape = traj.actions.shape

    if ac_shape[0] < max_num_actions:
        zeros = np.zeros([max_num_actions - im_shape[0] + 1, im_shape[1], im_shape[2], im_shape[3], im_shape[4]], dtype=np.uint8)
        traj.images = np.concatenate([traj.images, zeros])

        if len(ac_shape) > 1:
            zeros = np.zeros([max_num_actions - ac_shape[0], ac_shape[1]])
        else:
            zeros = np.zeros([max_num_actions - ac_shape[0]])
        traj.actions = np.concatenate([traj.actions, zeros])

    assert traj.images.shape[0] == max_num_actions + 1
    assert traj.actions.shape[0] == max_num_actions

    return traj


def get_pad_mask(action_len, max_num_actions):
    """
     create a 0/1 mask with 1 where there are images and 0 where there is padding
    :param action_len:  the number of actions in trajectory
    :param max_num_actions:  maximum number of actions allowed
    :return:
    """
    if action_len < max_num_actions:
        mask = np.concatenate([np.ones(action_len + 1), np.zeros(max_num_actions - action_len)])
    elif action_len == max_num_actions:
        mask = np.ones(max_num_actions + 1)
    else:
        raise ValueError

    assert mask.shape[0] == max_num_actions + 1

    return mask


class HDF5SaverBase():
    def __init__(self, save_dir, traj_per_file,
                 offset=0, split=(0.90, 0.05, 0.05), split_train_val_test=True):

        self.train_test_val_split = split
        self.split_train_val_test = split_train_val_test
        self.traj_per_file = traj_per_file
        self.traj_lists = [[], [], []]   # train val test lists
        self.save_dir = save_dir
        self.traj_count = offset
        # TODO make this create dataset_spec

    def save_hdf5(self, traj_list, prefix):
        savedir = self.save_dir + '/hdf5/{}'.format(prefix) if self.split_train_val_test else self.save_dir + '/hdf5'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self.traj_count += 1

        with h5py.File(savedir + '/traj_{}to{}'.format((self.traj_count  - 1)*self.traj_per_file,
                                                self.traj_count*self.traj_per_file) + '.h5', 'w') as F:

            F['traj_per_file'] = self.traj_per_file
            for i, traj in enumerate(traj_list):
                key = 'traj{}'.format(i)

                assert traj.images.dtype == np.uint8, 'image need to be uint8!'
                for name, value in traj.items():
                    F[key + '/' + name] = value

    def make_traj(self):
        raise NotImplementedError()

    def save_traj(self):
        raise NotImplementedError()

    def _save_traj(self, traj):
        draw = np.random.choice([0, 1, 2], 1, p=self.train_test_val_split)[0]
        self.traj_lists[draw].append(traj)

        for i, prefix in enumerate(['train', 'val', 'test']):
            if len(self.traj_lists[i]) == self.traj_per_file:
                self.save_hdf5(self.traj_lists[i], prefix)
                self.traj_lists[i] = []

    def make_dataset(self):
        boundaries = np.cumsum(np.array(self.train_test_val_split) * len(self.filenames), 0).astype(int)

        self.make_phase(self.filenames[:boundaries[0]], 'train')

        self.make_phase(self.filenames[boundaries[0]:boundaries[1]], 'val')

        self.make_phase(self.filenames[boundaries[1]:], 'test')


class HDF5Saver(HDF5SaverBase):
    def __init__(self, save_dir, envparams, agentparams, traj_per_file,
                 offset=0, split=(0.90, 0.05, 0.05), split_train_val_test=True):

        if hasattr(agentparams, 'max_num_actions'):
            self.max_num_actions = envparams.max_num_actions
        else:
            self.max_num_actions = agentparams.T

        super().__init__(save_dir, traj_per_file, offset, split, split_train_val_test)

    def _save_manifests(self, agent_data, obs, policy_out):
        pass

    def make_traj(self, obs, policy_out):
        traj = AttrDict()

        traj.images = obs['images']
        traj.states = obs['state']
        
        action_list = [action['actions'] for action in policy_out]
        traj.actions = np.stack(action_list, 0)
        
        traj.pad_mask = get_pad_mask(traj.actions.shape[0], self.max_num_actions)
        traj = pad_traj_timesteps(traj, self.max_num_actions)
    
        return traj

    def save_traj(self, itr, agent_data, obs, policy_out):
        traj = self.make_traj(obs, policy_out)
        self._save_traj(traj)


