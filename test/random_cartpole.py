import gym
import numpy as np
import matplotlib.pyplot as plt


def run_episode(env, parameters):
    #obtain observation
    observation = env.reset()
    #init total reward value
    totalreward = 0
    #require keep straight in 200 timestep(one episode)
    for _ in range(200):
        #use liner combiration to select action by inner product,less than 0,move left;otherwise,move right.0=letf,1=right
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        #obtain ob,reward,done after action
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

def train(submit):
    env = gym.make('CartPole-v0')

    counter = 0
    bestparams = None
    bestreward = 0
    for _ in range(10000):
        counter += 1
        #stochastic initializing between[-1,1]
        parameters = np.random.rand(4) * 2 - 1
        noise_scale=0.1
        parameters=parameters+(np.random.rand(4) * 2 - 1)*noise_scale
        #parameters = parameters + parameters * noise_scale
        reward = run_episode(env,parameters)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            if reward == 200:
                break
    return counter

# train an agent to submit to openai gym
# train(submit=True)

# create graphs
results = []
for _ in range(1000):
    results.append(train(submit=False))

plt.hist(results,50,density=1, facecolor='g', alpha=0.75)
plt.xlabel('Episodes required to reach 200')
plt.ylabel('Frequency')
plt.title('Histogram of Random Search')
plt.show()

print (np.sum(results) / 1000.0)