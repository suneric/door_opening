""" 
A PPO agent class using image as input
"""
import rospy
import tensorflow as tf
import numpy as np
import scipy.signal
import tensorflow_probability as tfp
tfd = tfp.distributions

################################################################
"""
Can safely ignore this block
"""
# restrict GPU and memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
################################################################


################################################################
"""
On-policy Replay Buffer for PPO
"""
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:

    def __init__(self, dim_obs, dim_act, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros([size]+list(dim_obs), dtype=np.float32)
        # self.obs_buf = np.zeros((size, dim_obs), dtype=np.float32)
        self.act_buf = np.zeros((size, dim_act), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr
        # self.ptr, self.path_start_idx = 0, 0

    def get(self):
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in data.items()}
################################################################

def convnet(dim_inputs, dim_outputs, activation, output_activation=None):
    # inputs
    img_inputs = tf.keras.Input(shape=dim_inputs, name='img_inputs')
    # image features
    img_feature = tf.keras.layers.Conv2D(32,(3,3), padding='same', activation=activation)(img_inputs)
    img_feature = tf.keras.layers.MaxPool2D((2,2))(img_feature)
    img_feature = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=activation)(img_feature)
    img_feature = tf.keras.layers.MaxPool2D((2,2))(img_feature)
    img_feature = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=activation)(img_feature)
    img_feature = tf.keras.layers.Flatten()(img_feature)
    img_feature = tf.keras.layers.Dense(128, activation=activation)(img_feature)
    # outputs
    outputs = tf.keras.layers.Dense(dim_outputs, activation=output_activation)(img_feature)

    return tf.keras.Model(inputs=img_inputs, outputs=outputs)

class CategoricalActor(tf.keras.Model):

    def __init__(self, dim_obs, dim_act, **kwargs):
        super(CategoricalActor, self).__init__(name='categorical_actor', **kwargs)
        self.logits_net = convnet(dim_inputs=dim_obs, dim_outputs=dim_act, activation='tanh')

    def _distribution(self, obs):
        logits = self.logits_net(obs)

        return tfd.Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def call(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, np.squeeze(act))

        return pi, logp_a

class GaussianActor(tf.keras.Model):

    def __init__(self, dim_obs, dim_act, **kwargs):
        super(GaussianActor, self).__init__(name='gaussian_actor', **kwargs)
        self.log_std = tf.Variable(initial_value=-0.5*np.ones(dim_act, dtype=np.float32))
        self.mu_net = convnet(dim_inputs=dim_obs, dim_outputs=dim_act, activation='relu')

    def _distribution(self, obs):
        mu = tf.squeeze(self.mu_net(obs))
        std = tf.math.exp(self.log_std)

        return tfd.Normal(loc=mu, scale=std)

    def _log_prob_from_distribution(self, pi, act):
        return tf.math.reduce_sum(pi.log_prob(act), axis=-1)

    def call(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, logp_a
    
class Critic(tf.keras.Model):

    def __init__(self, dim_obs, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        self.val_net = convnet(dim_inputs=dim_obs, dim_outputs=1, activation='relu')

    @tf.function
    def call(self, obs):
        return tf.squeeze(self.val_net(obs), axis=-1)

class PPOAgent(tf.keras.Model):

    def __init__(self, env_type, dim_obs, dim_act, clip_ratio=0.2, lr_actor=1e-4,
                 lr_critic=3e-4, beta=0., target_kl=0.01, **kwargs):
        super(PPOAgent, self).__init__(name='ppo', **kwargs)
        self.env_type = env_type
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        if env_type == 'discrete':
            self.actor = CategoricalActor(dim_obs, dim_act)
        elif env_type == 'continuous':
            self.actor = GaussianActor(dim_obs, dim_act)
        self.critic = Critic(dim_obs)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.actor_loss_metric = tf.keras.metrics.Mean()
        self.critic_loss_metric = tf.keras.metrics.Mean()
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.target_kl = target_kl

    def pi_of_a_given_s(self, obs):
        with tf.GradientTape() as t:
            with t.stop_recording():
                pi = self.actor._distribution(obs) # policy distribution (Gaussian)
                act = tf.squeeze(pi.sample())
                logp_a = self.actor._log_prob_from_distribution(pi, act)
                val = tf.squeeze(self.critic(obs), axis=-1)

        return act.numpy(), val.numpy(), logp_a.numpy()

    def train(self, data, iter_a, iter_c):
        # update actor
        for i in range(iter_a):
            rospy.logdebug("Staring actor epoch: {}".format(i+1))
            ep_kl = tf.convert_to_tensor([]) 
            ep_ent = tf.convert_to_tensor([]) 
            with tf.GradientTape() as tape:
                tape.watch(self.actor.trainable_variables)
                pi, logp = self.actor(data['obs'], data['act']) 
                ratio = tf.math.exp(logp - data['logp']) # pi/old_pi
                clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio), data['adv'])
                approx_kl = tf.reshape(data['logp'] - logp, shape=[-1])
                ent = tf.reshape(tf.math.reduce_sum(pi.entropy(), axis=-1), shape=[-1])
                obj = tf.math.minimum(tf.math.multiply(ratio, data['adv']), clip_adv) + self.beta*ent
                loss_pi = -tf.math.reduce_mean(obj)
            # gradient descent actor weights
            grads_actor = tape.gradient(loss_pi, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
            # record kl-divergence and entropy
            ep_kl = tf.concat([ep_kl, approx_kl], axis=0)
            ep_ent = tf.concat([ep_ent, ent], axis=0)
            # log epoch
            kl = tf.math.reduce_mean(ep_kl)
            entropy = tf.math.reduce_mean(ep_ent)
            print("Epoch :{} \nLoss: {} \nEntropy: {} \nKLDivergence: {}".format(
                i+1,
                loss_pi,
                entropy,
                kl
            ))
            # early cutoff due to large kl-divergence
            if kl > 1.5*self.target_kl:
                rospy.logwarn("Early stopping at epoch {} due to reaching max kl-divergence.".format(i+1))
                break
        # update critic
        for i in range(iter_c):
            rospy.logdebug("Starting critic epoch: {}".format(i))
            with tf.GradientTape() as tape:
                tape.watch(self.critic.trainable_variables)
                loss_v = tf.keras.losses.MSE(data['ret'], self.critic(data['obs']))
            # gradient descent critic weights
            grads_critic = tape.gradient(loss_v, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
            # log epoch
            print("Epoch :{} \nLoss: {}".format(
                i+1,
                loss_v
            ))

        return loss_pi, loss_v, dict(kl=kl, ent=entropy) 
