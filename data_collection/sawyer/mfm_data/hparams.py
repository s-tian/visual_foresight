""" Hyperparameters for Large Scale Data Collection (LSDC) """

import os.path
from visual_mpc.policy.random.gaussian import GaussianPolicy
from visual_mpc.agent.general_agent import GeneralAgent
from visual_mpc.envs.robot_envs.autograsp_env import AutograspEnv 
from visual_mpc.envs.robot_envs.util.topic_utils import IMTopic


BASE_DIR = '/'.join(str.split(__file__, '/')[:-1])
current_dir = os.path.dirname(os.path.realpath(__file__))

env_params = {
    'upper_bound_delta': [0.07, 0., 0., 0., 0.],
    'lower_bound_delta': [0.07, 0., 0, 0., 0.],
    'gripper_attached': 'sawyer_gripper',
    'camera_topics': [IMTopic('/front/image_raw')],
    #'cleanup_rate': 1,
}


agent = {
    'type': GeneralAgent,
    'env': (AutograspEnv, env_params),
    'data_save_dir': BASE_DIR,
    'T': 30,
    'image_height' : 480,
    'image_width' : 640,
    'record': BASE_DIR + '/record/',
}


policy = {
    'type': GaussianPolicy,
    'nactions': 10,
    'initial_std': 0.035,   #std dev. in xy
    'initial_std_lift': 0.08,   #std dev. in z
}


config = {
    'traj_per_file':128,
    'current_dir' : current_dir,
    'save_data': True,
    'save_raw_images': True,
    'start_index':0,
    'end_index': 120000,
    'agent': agent,
    'policy': policy,
    'ngroup': 1000
}
