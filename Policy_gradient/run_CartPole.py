"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL import PolicyGradient

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

first_achieve=0
first_flag=True
achieve_count=0
achieve_mean=0

for i_episode in range(1000):

    observation = env.reset()
    timestep=0
    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done or timestep == 200:
            if (not done) or (done and timestep == 200):
                if first_flag:
                    first_flag=False
                    first_achieve=i_episode
                achieve_count += 1
                achieve_mean = achieve_mean + i_episode

            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            loss = RL.learn()

            break

        observation = observation_
        timestep+=1

print("first achieve: %d,achieve reward count: %d,average episode: %g" % (first_achieve,achieve_count,achieve_mean/achieve_count))
RL.plot_cost()
