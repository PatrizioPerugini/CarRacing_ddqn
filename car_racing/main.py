import argparse
import random
import numpy as np
from student import Policy
import gym


def evaluate(env=None, n_episodes=2, render=False):
    agent = Policy()
    agent.load()

    env = gym.make('CarRacing-v2', continuous=agent.continuous)
    if render:
        env = gym.make('CarRacing-v2', continuous=agent.continuous, render_mode='human')
        
    rewards = []
    #ADDED THIS BECAUSE NOT ASSIGNED
    max_steps_per_episode = 1000
    for episode in range(n_episodes):
        
        env = gym.make('CarRacing-v2', continuous=agent.continuous,render_mode='human')
        print("--------------------------")
        print("starting episode number",episode)
        total_reward = 0
        done = False
        s, _ = env.reset()
        for i in range(max_steps_per_episode):
            action = agent.act(s)
            s, reward, done, truncated, info = env.step(action)
            if render: env.render()
            total_reward += reward
            if done or truncated: break
        print("total reward for this episode is", total_reward)
        rewards.append(total_reward)
        
    print('Mean Reward:', np.mean(rewards))



def train():
    agent = Policy()
    agent.train()
    agent.save()


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()
        
    if args.evaluate:
        evaluate(render=args.render)

    
if __name__ == '__main__':
    #to run: python3 main.py --render -e
    main()
