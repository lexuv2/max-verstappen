from mountaincar_dqn import select_action, policy_net, device
import sys
import torch
import gymnasium as gym
from itertools import count
import numpy as np

if len(sys.argv) < 2:
    print("No model file provided")
    sys.exit(1)

model_name = sys.argv[1]
print(f"Loading model from {model_name}")

coefs = np.load(model_name)


test_count = 1000
results = []
finishes = 0

env = gym.make('MountainCar-v0')


def select_action(position, velocity, coefs):
    val = np.polynomial.polynomial.polyval2d([position], [velocity], coefs)
    return 2 if val > 0 else 0

for i in range(test_count):
    state, _ = env.reset()
    position, velocity = state
    for t in count():
        action = select_action(position,velocity,coefs)
        observation, _, terminated, truncated, _ = env.step(action)
        position, velocity = observation
        if terminated or truncated:
            results.append(t + 1)
            if terminated:
                finishes += 1
            break

env.close()

finished = len(results)

print(f"Average duration: {sum(results) / test_count}")
print(f"Min duration: {min(results)}")
print(f"Finsihed: {finishes}/{finished} ({finishes / finished * 100:.2f}%)")
