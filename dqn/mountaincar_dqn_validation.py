from mountaincar_dqn import select_action, policy_net, device
import sys
import torch
import gymnasium as gym
from itertools import count

if len(sys.argv) < 2:
    print("No model file provided")
    sys.exit(1)

model_name = sys.argv[1]
print(f"Loading model from {model_name}")

policy_net.load_state_dict(torch.load(sys.argv[1]))

test_count = 1000
results = []
finishes = 0

env = gym.make('MountainCar-v0')

for i in range(test_count):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, _, terminated, truncated, _ = env.step(action.item())

        if terminated or truncated:
            results.append(t + 1)
            if terminated:
                finishes += 1
            break

        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

env.close()

finished = len(results)

print(f"Average duration: {sum(results) / test_count}")
print(f"Min duration: {min(results)}")
print(f"Finsihed: {finishes}/{finished} ({finishes / finished * 100:.2f}%)")
