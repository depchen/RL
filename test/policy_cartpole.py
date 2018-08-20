import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt

#policy gradient core,main function:maximize probability,get optimal params
def policy_gradient():
    with tf.variable_scope("policy"):
        #define variable,init param
        params = tf.get_variable("policy_parameters",[4,2])
        state = tf.placeholder("float",[None,4])
        actions = tf.placeholder("float",[None,2])
        #how much better was this action than normal
        advantages = tf.placeholder("float",[None,1])
        #output two action probabilities
        linear = tf.matmul(state,params)
        probabilities = tf.nn.softmax(linear)
        #optimizer
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions),reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probabilities, state, actions, advantages, optimizer

#opmital network params to calculate current reward,
def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None,4])
        newvals = tf.placeholder("float",[None,1])
        w1 = tf.get_variable("w1",[4,10])
        b1 = tf.get_variable("b1",[10])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[10,1])
        b2 = tf.get_variable("b2",[1])
        #current reward
        calculated = tf.matmul(h1,w2) + b2
        #calculate diff between current and future
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss

#include two part:one is execute reinforce learning,two is calculate advantage and future reward
def run_episode(env, policy_grad, value_grad, sess):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

    #execute reinforcelearn and record transitions,transition contain action,
    for _ in range(200):
        # obtain observation data
        obs_vector = np.expand_dims(observation, axis=0)
        #calculate probs
        probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
        #agent confirm action
        action = 0 if random.uniform(0,1) < probs[0][0] else 1
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        #execute action
        observation, reward, done, info = env.step(action)
        #record transition
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
    # calculate discounted monte-carlo return,get future reward and
    for index, trans in enumerate(transitions):
        obs, action, reward = trans
        future_reward = 0
        future_transitions = len(transitions) - index
        #discount coefficient,future reward exist uncertainty,prevent reward infinite increase
        decrease = 1
        #calculate future reward,
        for index2 in range(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            #decrease gradually taper,
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})[0][0]

        # advantage: how much better was this action than normal
        advantages.append(future_reward - currentval)

        # update the value function towards new return
        update_vals.append(future_reward)

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})
    return totalreward

policy_grad = policy_gradient()
value_grad = value_gradient()
def train():
    counter=0
    env = gym.make('CartPole-v0')
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(2000):
        counter=counter+1
        reward = run_episode(env, policy_grad, value_grad, sess)
        env.render()
        if reward == 200:
            # print ("reward 200")
            # print (i)
            break
    return counter


results=[]
t = 0
for _ in range(500):
    results.append(train())

plt.hist(results,50,density=1, facecolor='g', alpha=0.75)
plt.xlabel('Episodes required to reach 200')
plt.ylabel('Frequency')
plt.title('Histogram of Random Search')
plt.show()

print (np.sum(results) / 500.0)