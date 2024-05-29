import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

def plot_duration(num_episodes: int = 0, rewards: list[int] = 0) -> None:
    plt.figure(1, figsize=(10, 5))
    plt.clf()
    plt.title(f'Training... ({num_episodes}/{len(rewards)})')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards[:num_episodes], '.')

    plt.pause(0.05)

def run(episodes, is_training=True, render=False):

    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)    # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)    # Between -0.07 and 0.07

    if(is_training):
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # init a 20x20x3 array
    else:
        model_name = f"models/{episodes}_mountain_car.pkl"
        print(f"Loading model from {model_name}")
        f = open(model_name, 'rb')
        q = pickle.load(f)
        f.close()

        # when testing, set episodes to 1000 (i.e. 1000 games)
        episodes = 1000

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor.

    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 2/episodes # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False          # True when reached goal

        rewards=0

        while(not terminated and rewards>-1000):

            if is_training and rng.random() < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])

            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + learning_rate_a * (
                    reward + discount_factor_g*np.max(q[new_state_p, new_state_v,:]) - q[state_p, state_v, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            rewards+=reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards
        if is_training:
            print(f'Episode {i+1} finished after with rewards {rewards}, terminated: {terminated}')
            plot_duration(i+1, rewards_per_episode)

    env.close()

    # Save Q table to file
    if is_training:
        f = open(f'models/{episodes}_mountain_car.pkl','wb')
        pickle.dump(q, f)
        f.close()

        mean_rewards = np.zeros(episodes)
        for t in range(episodes):
            mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(mean_rewards)
        plt.savefig(f'plots/{episodes}_episodes.png')
    else:
        num_finished = np.sum(rewards_per_episode >= -200)
        print(f"Average duration: {np.mean(rewards_per_episode)}")
        print(f"Max duration: {np.max(rewards_per_episode)}")
        print(f"Finished: {num_finished}/{episodes} ({num_finished/episodes*100:.2f}%)")

def run_training():
    episodes = [250, 500, 1000, 2500]    #, 5000]
    for i in episodes:
        run(i, is_training=True, render=False)

def run_validation():
    to_test = [250, 500, 1000, 2500, 5000]
    for i in to_test:
        print(f"Testing model trained with {i} episodes")
        run(i, is_training=False, render=False)

if __name__ == '__main__':
    if len(sys.argv) > 1 and "train" in sys.argv[1]:
        run_training()
    else:
        run_validation()
    #run(1000, is_training=False, render=True)
