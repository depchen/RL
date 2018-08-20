import gym
from RL import RL_brain

env=gym.make('CartPole-v0')
env = env.unwrapped
# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

RL=RL_brain.DeepQNetwork(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.01,
    reward_decay=0.9,
    e_greedy=0.9,
    replace_target_iter=100,
    memory_size=2000,
    batch_size=32,
    e_greedy_increment=0.001,
)
total_step=0

for episode in range(100):
    total_reward = 0

    observation=env.reset()
    while True:
        env.render()
        total_step+=1
        action=RL.choose_action(observation)
        obs,r,done,info=env.step(action)
        x, x_dot, theta, theta_dot = obs
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        RL.store_transition(s=observation,a=action,r=reward,s_=obs)
        total_reward+=reward
        if total_step>1000:
            RL.learn()
        if done:
            print('episode: ', episode,
                  'ep_r: ', round(total_reward, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break
        observation=obs
RL.plot_cost()