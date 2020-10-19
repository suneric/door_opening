#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tensorflow as tf

from agents.ppo_conv import PPOAgent
from envs.door_open_specific_envs import DoorPullTaskEnv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise' type=float, default=0.0)
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    rospy.init_node('ppo_test', anonymous=True, log_level=rospy.INFO)
    # load model
    env = DoorPullTaskEnv(resolution=(64,64),cam_noise=args.noise)
    dim_obs = (64,64,3)
    dim_act = env.action_dimension()
    agent = PPOAgent(
        env_type='discrete',
        dim_obs=dim_obs,
        dim_act=dim_act,
    )
    logits_net_path = './saved_models/door_open/ppovar.05/logits_net/49'
    agent.actor.logits_net = tf.keras.models.load_model(logits_net_path)
    success_counter = 0

    # start test
    for ep in range(10):
        obs, info = env.reset()
        done = False
        for st in range(60):
            # obs += np.clip(np.random.normal(loc=0., scale=1, size=obs.shape), -.5, .5)
            act, _, _ = agent.pi_of_a_given_s(np.expand_dims(obs, axis=0))
            n_obs, rew, done, info = env.step(act)
            obs = n_obs.copy()
            if done:
                if env.success:
                    success_counter += 1
                print("Succeeded: {}".format(success_counter))
                break
