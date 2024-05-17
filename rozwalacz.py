import gymnasium as gym
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
poly_deg = 1
pop_size = 500
parents_size = 20
mutation_percent = 30
threads = 8
threading=True #not properly implemented
save_best = True


game_name = "MountainCar-v0"



gui = False



def select_action(position, velocity, coefs):
    val = np.polynomial.polynomial.polyval2d([position], [velocity], coefs)
    #print(f'Value: {val}')
    return 2 if val > 0 else 0
    
def r():
    return np.random.uniform(-1,1)

def play_game(coefs,seed: int = None,):
    env_gui = gym.make(game_name, render_mode='human' if gui else None)
    state, _ = env_gui.reset(seed=seed)
    position, velocity = state
    last_position = position
    speed_sum = 0
    for t in count():
        action = select_action(position,velocity,coefs)
        #print(f'Action: {action}')
        observation, _, terminated, truncated, _ = env_gui.step(action)
        position, velocity = observation
        #print(f'Position: {position}, Velocity: {velocity}')
        speed_sum += abs(position-last_position)
        last_position = position
        if terminated or truncated:
            break
    env_gui.close()
    if t < 199:
        print(f'Terminated: {terminated}, Truncated: {truncated}, Steps: {t}')
        return ((200/t),coefs)
    return (speed_sum/t,coefs)

def random_coefs():
    return [[r() for _ in range(poly_deg+1)] for _ in range(poly_deg+1)]

def empty_coefs():
    return [[0 for _ in range(poly_deg+1)] for _ in range(poly_deg+1)]

population = [random_coefs() for _ in range(pop_size)]
plt.ion()
epoch = 0
best_of_best = 0
best_of_best_coefs = []
pool = Pool(threads)
while True:
    epoch += 1
    results = []
    best_fit = 0
    best_coefs = []
    if not threading:
        for x in population:
            fit,_ = play_game(x)
            if fit > best_fit:
                best_fit = fit
                best_coefs = x
            if fit > best_of_best:
                best_of_best = fit
                best_of_best_coefs = x
            results.append((x,fit)) 
    else:
        
        for result in pool.map(play_game, population):
            fit,x = result
            if fit > best_fit:
                best_fit = fit
                best_coefs = x
            if fit > best_of_best:
                best_of_best = fit
                best_of_best_coefs = x
            results.append((x,fit))
    
    probabilitites = [x[1] for x in results]
    probabilitites = [x/sum(probabilitites) for x in probabilitites]
    parents_indexes = np.random.choice(range(0,len(results)), size=parents_size, p=probabilitites)
    parents = [results[x][0] for x in parents_indexes]
    population = []
    for _ in range(pop_size):
        first_parent = parents[np.random.randint(0,parents_size)]
        second_parent = parents[np.random.randint(0,parents_size)]
        child = empty_coefs()
        for i in range(poly_deg+1):
            for j in range(poly_deg+1):
                #child[i][j] = (first_parent[i][j] + second_parent[i][j])/2
                if r() < 0.5:
                    child[i][j] = first_parent[i][j]
                else:
                    child[i][j] = second_parent[i][j]
        population.append(child)

    #mutation
    for x in population:
        for i in range(poly_deg+1):
            for j in range(poly_deg+1):
                x[i][j] += (r()/100)*mutation_percent
    print(f'Best fit: {best_fit}')
    print(f'Best coefs: {best_coefs}')
    print("Best tested: ")
    env_gui = gym.make(game_name, render_mode=None)
    play_game(best_coefs)

    plt.plot(epoch,best_fit,'ro')
    plt.pause(0.05)
    ##replay the best
    if save_best and (epoch % 10) == 0:
        np.save(f"best_iwicki_polynomial{best_of_best}.npy",best_of_best_coefs)
    if (epoch % 10) == 0:
        env_gui = gym.make(game_name, render_mode='human')
        print("TESTING BEST:")
        play_game(best_of_best_coefs)
        env_gui = gym.make(game_name, render_mode=None)
    


    
    