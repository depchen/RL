"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time


np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 1000
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 500
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = False
ENV_NAME = 'CartPole-v0'

###############################  Actor  ####################################


class Actor(object):
    def __init__(self, sess, action_dim, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        #self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(
                s,
                30,
                activation=tf.nn.relu,
                kernel_initializer=init_w,
                bias_initializer=init_b,
                name='l1',
                trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(
                    net,
                    self.a_dim,
                    activation=tf.nn.softmax,#-1 - a - 1
                    kernel_initializer=init_w,
                    bias_initializer=init_b,
                    name='a',
                    trainable=trainable)
                #actions=np.argmax(actions)
                #scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return actions

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def return_action(self,s):
        s = s[np.newaxis, :]
        actions = self.sess.run(self.a, feed_dict={S: s})
        return actions

    def choose_action(self, s):
        s = s[np.newaxis, :]
        action = self.sess.run(self.a, feed_dict={S: s})  # get probabilities for all actions
        action = np.argmax(action)
        return action

        # s = s[np.newaxis, :]    # single state
        # b=self.sess.run(self.a,feed_dict={S: s})
        # a=self.sess.run(self.a,feed_dict={S: s})[0]
        # return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams

            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

            #self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                # l1 = tf.layers.dense(
                #     inputs=s,
                #     units=20,  # number of hidden units
                #     activation=tf.nn.relu,  # None
                #     # have to be linear to make sure the convergence of actor.
                #     # But linear approximator seems hardly learns the correct Q.
                #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                #     bias_initializer=tf.constant_initializer(0.1),  # biases
                #     name='l1'
                # )
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                # v = tf.layers.dense(
                #     inputs=l1,
                #     units=1,  # output units
                #     activation=None,
                #     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                #     bias_initializer=tf.constant_initializer(0.1),  # biases
                #     name='V'
                # )
        # return v
                q = tf.layers.dense(
                    net,
                    1,
                    kernel_initializer=init_w,
                    bias_initializer=init_b,
                    trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        _,loss=self.sess.run([self.train_op,self.loss], feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a0,a1, r, s_):
        transition = np.hstack((s, [a0,a1, r], s_))
        #transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        # sample batch memory from all memory
        if self.pointer > self.capacity:
            sample_index = np.random.choice(self.capacity, size=BATCH_SIZE)
        else:
            sample_index = np.random.choice(self.pointer, size=BATCH_SIZE)
        batch_memory = self.data[sample_index, :]
        return batch_memory

        # assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        # indices = np.random.choice(self.capacity, size=n)
        # return self.data[indices, :]


env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
#action_bound = env.action_space.high

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None,], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')


sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
#actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
actor = Actor(sess, action_dim, LR_A, REPLACEMENT)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + 3)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

var = 3  # control exploration

first_achieve=0
first_flag=True
achieve_count=0
achieve_mean=0

losses=[]

t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    for j in range(MAX_EP_STEPS):

        if RENDER:
            env.render()

        # Add exploration noise
        a = actor.choose_action(s)
        actions=actor.return_action(s)
        #a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)



        if M.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            b_M = M.sample(1)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim:state_dim+2]
            b_r = b_M[:, state_dim+3]
            b_s_ = b_M[:, -state_dim:]

            loss=critic.learn(b_s, b_a, b_r, b_s_)
            losses.append(loss)
            actor.learn(b_s)

        x, x_dot, theta, theta_dot = s_

        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        M.store_transition(s, actions[0][0],actions[0][1], reward, s_)

        s = s_
        ep_reward += reward

        if j == MAX_EP_STEPS-1 or done:
            if (not done) or (done and j == MAX_EP_STEPS-1):
                if first_flag:
                    first_flag=False
                    first_achieve=i
                achieve_count += 1
                achieve_mean = achieve_mean + i

            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:
                RENDER = True
            break

print("actor critic first achieve: %d,achieve reward count: %d,average episode: %g" % (first_achieve,achieve_count,achieve_mean/achieve_count))

import matplotlib.pyplot as plt
plt.plot(np.arange(len(losses)), losses)
plt.ylabel('Cost')
plt.xlabel('training steps')
plt.show()
print('Running time: ', time.time()-t1)