"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from RL import DeepQNetwork

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0
first_achieve=0
first_flag=True
achieve_count=0
achieve_mean=0

for i_episode in range(1000):

    observation = env.reset()
    ep_r = 0
    timestep=0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if timestep == 200 and first_flag:
            first_achieve=i_episode
            first_flag=False
            achieve_count+=1
            achieve_mean=achieve_mean+i_episode
            break
        if timestep == 200 or done:
            if done:
                print('episode: ', i_episode,
                      'ep_r: ', round(ep_r, 2),
                      ' epsilon: ', round(RL.epsilon, 2))
                break
            else:
                achieve_count+=1
                achieve_mean=achieve_mean+i_episode
                break

        observation = observation_
        total_steps += 1
        timestep+=1
print("first achieve: %d,achieve reward count: %d,average episode: %g" % (first_achieve,achieve_count,achieve_mean/achieve_count))
RL.plot_cost()
