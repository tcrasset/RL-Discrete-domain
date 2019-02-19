import math
import numpy as np
import random
import os.path

# CONSTANTS
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
ACTIONS = [UP, DOWN, LEFT, RIGHT]


def expectedReward(current, action, reward_matrix, beta=0):
    """
    Computes the expected reward of a state.
    The beta paramter specifies the threshold for the
    stochastic dynamics, if none is given, it defaults to 0, 
    that is a purely deterministic approach.

    Arguments:
    ---------
    - current: Tuple containing the coordinates of the current state
    - action: Tuple containing the action to take in the current state 
    - reward_matrix: matrix containing the rewards of each cell
    - beta: threshold for stochastic dynamics

    Returns:
    --------
    Integer value fo the expected reward of the cell
    """
    coords = dynamics(current, 
                    action, 
                    reward_matrix.shape[0], 
                    reward_matrix.shape[1])

    deterministic = rewardAtPosition(reward_matrix, coords)
    stochastic = rewardAtPosition(reward_matrix, (0, 0))
    return (1 - beta) * deterministic + beta * stochastic

def probability(current, action, test_state, reward_matrix, beta):
    """
    Computes the probability of reaching a next state
    by taking the given action in the current state. 
    Arguments:
    ---------
    - current: Tuple containing the coordinates of the current state
    - action: Tuple containing the action to take in the current state 
    - test_state: Tuple containing the coordiantes of the next state to test
    - reward_matrix: matrix containing the rewards for each cell
    - beta: threshold for stochastic dynamics
    Returns:
    --------
    Float value between 0 and 1
    """

    dim_x = reward_matrix.shape[0]
    dim_y =reward_matrix.shape[1]
    next_state = dynamics(current, action, dim_x, dim_y)

    if beta == 0: # Deterministic
        if next_state == test_state:
            return 1 
        else:
            return 0
    else: # Stochastic
        if test_state == (0,0):
            return beta
        elif next_state == test_state:
            return (1 -beta)  
        else:
             return 0


def rewardAtPosition(reward_matrix, coords):
    """Returns the reward from a given cell in reward_matrix
    whose coordinates are given by a tuple class
    Arguments:
    ----------
    - reward_matrix: matrix containing the rewards for each cell
    - coords : tuple describing the cell from which to extract the reward
    Returns:
    ----------
    - integer reward
    """
    return reward_matrix[coords]


def dynamics(curr, action, dim_x, dim_y):
    """
    Returns the next state given the current state
    and the action to perform
    Arguments:
    ----------
    - curr : tuple giving current state coordinates
    - action: tuple giving the action to take
    - dim_x: upper bound of the first dimension
    - dim_y: upper bound of the second dimension
    Returns:
    --------
    Tuple corresponding to the coordinates of the next state
    """
    maximum_x = max(curr[0] + action[0], 0)
    maximum_y = max(curr[1] + action[1], 0)
    return (min(maximum_x, dim_x - 1), min(maximum_y, dim_y - 1))


def computeJ(policy, reward_matrix, discount, N, beta=0):
    """
    Computes the cumulative expected reward for a whole grid
    Add the beta parameter to use a stochastic approach
    Without it, a deterministic approach is chosen.
    Arguments:
    ----------
    - policy: dictionnary containing the action to take at each state
    - reward_matrix: matrix containing the rewards for each cell
    - discount : discount value between each iteration
    - N : number of iterations to perform
    - beta: probability threshold for stochastic dynamics
    Returns:
    ----------
    - Matrix with the same shape as reward_matrix containing
    in each cell the cumulative expected reward of that state 
    """
    dim_x = reward_matrix.shape[0]
    dim_y = reward_matrix.shape[1]
    current_J = np.zeros_like(reward_matrix, dtype=float)
    for i in range(0, N):
        previous_J = current_J

        for x in range(dim_x):
            for y in range(dim_y):
                next_point = dynamics((x, y), policy[(x,y)], dim_x, dim_y)
                current_J[(x, y)] = expectedReward((x,y), policy[(x,y)], reward_matrix, beta) \
                    + discount * \
                    ((1 - beta) *
                     previous_J[next_point] + beta * previous_J[(0, 0)])
    return current_J


def computeProba(action, reward_matrix, beta):
    """
    Compute the probability to reach any next state from any state
    using the given action

    Arguments:
    ----------
    - action: tuple giving the next action to take
    - reward_matrix: matrix containing the rewards for each cell
    - beta: probability threshold for stochastic dynamics
    Returns:
    ----------
    - Dictionnary with current_state as key and probability grids
    giving the probability of reaching any next state 
    """
    dim_x = reward_matrix.shape[0]
    dim_y = reward_matrix.shape[1]
    # Dictionnary storing probability grids for any current_state, next_state pair
    # after having taken action `action`
    # Contains dim_x * dim_y entries of (dim_x, dim_y) dimension matrices
    # Key is current state
    prob_for_next_states = {}
    for x in range(dim_x):
        for y in range(dim_y):
            curent_state = (x,y)

            # Compute the probability to reach any next state from the current state
            # and stores it in a matrix of shape (dim_x, dim_y)
            prob = np.zeros_like(reward_matrix, dtype=float)
            for next_x in range(dim_x):
                for next_y in range(dim_y):
                    next_state = (next_x, next_y)
                    prob[next_state] = probability(curent_state, action, next_state, reward_matrix, 0)
            prob_for_next_states[curent_state] = prob

    return prob_for_next_states

def computeQ(reward_matrix, discount, N, beta = 0):
    """
    Computes the cumulative expected reward for state action pairs
    Add the beta parameter to use a stochastic approach
    Without it, a deterministic approach is chosen.
    Arguments:
    ----------
    - reward_matrix: matrix containing the rewards for each cell
    - discount : discount value between each iteration
    - beta: probability threshold for stochastic dynamics
    - N : number of iterations to perform
    Returns:
    ----------
    - Dictionnary containing entries for each different action
    Keys are actions and values are cumulative expected reward matrices
    with the same shape as `reward_matrix`
    """
    current_Q_dict = {}
    for u in ACTIONS:
        current_Q_dict[u] = np.zeros_like(reward_matrix, dtype=float)

    # Compute the whole probability grid for every current_state , next_state pair
    # for a given action
    for it in range(N):
        print("N = {}".format(it))
        previous_Q_dict = current_Q_dict

        for current_x in range(reward_matrix.shape[0]):
            for current_y in range(reward_matrix.shape[1]):
                current_state = (current_x, current_y)
                for u in ACTIONS:
                    prob_dict = computeProba(u, reward_matrix, beta)
                    expectation = 0

                    # Compute the expectation of the reward of previous states
                    for next_x in range(reward_matrix.shape[0]):
                        for next_y in range(reward_matrix.shape[1]):
                            next_state = (next_x, next_y)

                            # Finding maximum cumulative reward of the previous iteration
                            max_reward = -math.inf
                            for u_prime in ACTIONS:
                                current_Q = current_Q_dict[u_prime]
                                if current_Q[next_state] > max_reward:
                                    max_reward = previous_Q_dict[u_prime][next_state]

                            expectation += prob_dict[current_state][next_state] * max_reward

                    reward = expectedReward(current_state, u, reward_matrix, beta)
                    current_Q_dict[u][current_state] = reward + discount * expectation
    return current_Q_dict


def createTrajectory(N, reward_matrix, beta=0):
    """
    Creates a trajectory h in the reward_matrix grid
    that is `N` states long, with random ACTIONS and a
    random starting state.
    h is of the form (state_0, action0, reward_0, ..., state_N)

    Arguments:
    ----------
    - N : length of the trajectory
    - reward_matrix: matrix containing the rewards for each cell
    - beta: probability threshold for stochastic dynamics

    Returns:
    ----------
    - List containing the trajectory
    """

    # Creating trajectory h from a list of ACTIONS and a starting state
    h_start = (random.randint(0,4), random.randint(0,4))
    h_ACTIONS = [random.choice(ACTIONS) for i in range(N)]

    h = list()

    # First iteration
    h.append(h_start)
    h.append(h_ACTIONS[0])
    h.append(rewardAtPosition(reward_matrix, h_start))

    # Successive iterations
    dim_x = reward_matrix.shape[0]
    dim_y = reward_matrix.shape[1]
    for u in h_ACTIONS[1:]:
        w = random.random()
        if(w <= 1- beta):
            next_state = dynamics(h[-3], u, dim_x, dim_y)
        else:
            next_state = (0,0)
        h.append(next_state)
        h.append(u)
        h.append(rewardAtPosition(reward_matrix, h[-2]))
    h = h[:-2]
    return h


def computeAverageReward(N, reward_matrix):
    """
    Computes a good approximation of the reward r(x,u) from the trajectory h

    Arguments:
    ----------
    - N : length of the trajectory
    - reward_matrix: matrix containing the rewards for each cell

    Returns:
    ----------
    - List containing the trajectory
    """
    nb_occurences = np.zeros_like(reward_matrix, dtype=int)
    reward_sum = np.zeros_like(reward_matrix, dtype=int)

    traj = createTrajectory(N, reward_matrix, beta)
    i = 0
    while i < (len(traj) -1):
        nb_occurences[traj[i]] += 1
        reward_sum[traj[i]] += traj[i + 2]  
        i = i + 3

    np.seterr(divide='ignore')
    return np.nan_to_num(np.divide(reward_sum, nb_occurences), False)


def computeAverageProbability(N, reward_matrix):
    """
    Computes a good approximation of the probability p(x'| x,u) from the trajectory h

    Arguments:
    ----------
    - N : length of the trajectory
    - reward_matrix: matrix containing the rewards for each cell

    Returns:
    ----------
    - List containing the trajectory
    """
    dim_x = reward_matrix.shape[0]
    dim_y = reward_matrix.shape[1]
    nb_times_from_curr_to_next = {}
    nb_times_through_curr = {}
    average_prob = {}

    for x in range(dim_x):
        for y in range(dim_y):
            nb_times_from_curr_to_next[(x,y)] = np.zeros_like(reward_matrix, dtype=int)
            nb_times_through_curr[(x,y)] = 0
    traj = createTrajectory(N, reward_matrix, beta)

    i = 0
    while i < (len(traj) - 1):
        nb_times_from_curr_to_next[traj[i]][traj[i + 3]] += 1
        nb_times_through_curr[traj[i]] += 1
        i = i + 3
        
    for x in range(dim_x):
        for y in range(dim_y):

            np.seterr(divide='ignore')
            average_prob[(x,y)] = np.nan_to_num(np.divide(nb_times_from_curr_to_next[(x,y)], nb_times_through_curr[(x,y)]), False)
    return average_prob



def computeQApproximation(average_prob, average_reward, reward_matrix, discount, N):
    """
    Computes the cumulative expected reward for state action pairs
    Add the beta parameter to use a stochastic approach
    Without it, a deterministic approach is chosen.
    Arguments:
    ----------
    - reward_matrix: matrix containing the rewards for each cell
    - discount : discount value between each iteration
    - N : number of iterations to perform
    Returns:
    ----------
    - Dictionnary containing entries for each different action
    Keys are actions and values are cumulative expected reward matrices
    with the same shape as `reward_matrix`
    """
    dim_x = reward_matrix.shape[0]
    dim_y = reward_matrix.shape[1]
    current_Q_dict = {}
    for u in ACTIONS:
        current_Q_dict[u] = np.zeros_like(reward_matrix, dtype=float)


    # Compute the whole probability grid for every current_state , next_state pair
    # for a given action
    for it in range(N):
        print("N = {}".format(it))
        previous_Q_dict = current_Q_dict

        for current_x in range(dim_x):
            for current_y in range(dim_y):
                current_state = (current_x, current_y)
                for u in ACTIONS:
                    expectation = 0

                    # Compute the expectation of the reward of previous states
                    for next_x in range(dim_x):
                        for next_y in range(dim_y):
                            next_state = (next_x, next_y)

                            # Finding maximum cumulative reward of the previous iteration
                            max_reward = -math.inf
                            for u_prime in ACTIONS:
                                current_Q = current_Q_dict[u_prime]
                                # print("Next state {} iteration {} action {} reward {}".format(next_state, it, u_prime, current_Q[next_state]))
                                if current_Q[next_state] > max_reward:
                                    max_reward = previous_Q_dict[u_prime][next_state]

                            expectation += average_prob[current_state][next_state] * max_reward

                    reward = average_reward[dynamics(current_state, u, dim_x, dim_y)]
                    current_Q_dict[u][current_state] = reward + discount * expectation
    return current_Q_dict

def getOptimalPolicy(Q):
    """
    Computes the optimal policy from the state action value function

    Arguments:
    ----------
    Q : state action value function to compute the optimal policy from
    Returns:
    --------
    Dictionnary with states as keys and actions as values
    """
    # Optimal policy from the reccurence equation Q
    optimal_policy = {}
    dim_x = Q[UP].shape[0]
    dim_y = Q[UP].shape[1]
    for x in range(dim_x):
        for y in range(dim_y):
            best_action = None
            max_reward = -math.inf
            for u in ACTIONS:
                reward = rewardAtPosition(Q[u], (x,y))
                if(reward > max_reward):
                    max_reward = reward
                    best_action = u
            optimal_policy[(x,y)] = best_action
    for i in optimal_policy:
        if optimal_policy[i] == UP:
            print("{} : UP".format(i))
        elif optimal_policy[i] == DOWN:
            print("{} : DOWN".format(i))
        elif optimal_policy[i] == LEFT:
            print("{} : LEFT".format(i))
        elif optimal_policy[i] == RIGHT:
            print("{} : RIGHT".format(i))
    return optimal_policy




if __name__ == '__main__':
    """ --------> y
        |
        |
        |
        |
        v
        x
    """
    reward_matrix = np.matrix([[-3, 1, -5, 0, 19],
                                [6, 3, 8, 9, 10],
                                [5, -8, 4, 1, -8],
                                [6, -9, 4, 19, -5],
                                [-20, -17, -4, -3, 9]])

    RECOMPUTE = True
    discount = 0.99
    beta = 0
    N = 10000

    """Expected return of a policy - QUESTION 3"""
    # dim_x = reward_matrix.shape[0]
    # dim_y = reward_matrix.shape[1]
    # policy = {}
    # for x in range(dim_x):
    #     for y in range(dim_y):
    #         policy[(x,y)] = RIGHT
    # print("Initial matrix\n", reward_matrix)
    # # Deterministic computeJ
    # print("Cumulative reward with deterministic dynamics, 10000 iterations :\n",
    # computeJ(policy, reward_matrix, discount, 10000))
    # # Stochastic computeJ
    # print("Cumulative expected reward with stochastic dynamics, beta = 0.5, 10000 iterations :\n",
    # computeJ(policy, reward_matrix, discount, 10000, 0.5))

    """Optimal policy - QUESTION 4"""

    # # Initialization
    # current_Q_dict = {}
    # for u in ACTIONS:
    #     current_Q_dict[u] = np.zeros_like(reward_matrix, dtype=float)

    # # Save Q matrices in a file so as to not recompute it everytime
    # if (os.path.exists('./Q_UP.npy') and not RECOMPUTE):
    #     print("Loading from file...")
    #     current_Q_dict[UP] = np.load('Q_UP.npy')
    #     current_Q_dict[DOWN] = np.load('Q_DOWN.npy')
    #     current_Q_dict[LEFT] = np.load('Q_LEFT.npy')
    #     current_Q_dict[RIGHT] = np.load('Q_RIGHT.npy')
    # else:
    #     current_Q_dict = computeQ(reward_matrix, discount, N, beta)
    #     np.save('Q_UP.npy', current_Q_dict[UP])
    #     np.save('Q_DOWN.npy', current_Q_dict[DOWN])
    #     np.save('Q_LEFT.npy', current_Q_dict[LEFT])
    #     np.save('Q_RIGHT.npy', current_Q_dict[RIGHT])

    # print("UP\n",current_Q_dict[UP])
    # print("DOWN\n",current_Q_dict[DOWN])
    # print("LEFT\n",current_Q_dict[LEFT])
    # print("RIGHT\n",current_Q_dict[UP])

    # optimal_policy = getOptimalPolicy(current_Q_dict)

    # print(computeJ(optimal_policy, reward_matrix, discount, 10000))


    """System identification - QUESTION 5"""

    # #Compute average probability and reward with a trajectory
    # # of length 100000
    # average_prob = {}
    # average_reward = {}
    # # Save average matrices in a file so as to not recompute it everytime
    # if (os.path.exists('./average_prob.npy') and not RECOMPUTE):
    #     #Load
    #     average_prob = np.load('./average_prob.npy')
    #     average_reward = np.load('./average_reward.npy')
    #     average_prob = average_prob.item()
    # else:
    #     average_prob = computeAverageProbability(100000, reward_matrix)
    #     average_reward = computeAverageReward(100000, reward_matrix)
    #     # Save
    #     np.save('average_prob.npy', average_prob) 
    #     np.save('average_reward.npy', average_reward) 

    # # Printing
    # for i in average_prob:
    #     print("Starting from {}:\n{}".format(i,average_prob[i]))
    # print(average_reward)

    # # Init
    # current_Q_dict_approx = {}
    # for u in ACTIONS:
    #     current_Q_dict_approx[u] = np.zeros_like(reward_matrix, dtype=float)

    # # Save Q matrices in a file so as to not recompute it everytime
    # if (os.path.exists('./Q_UP_approx.npy') and not RECOMPUTE):
    #     #LOAD
    #     current_Q_dict_approx[UP] = np.load('Q_UP_approx.npy')
    #     current_Q_dict_approx[DOWN] = np.load('Q_DOWN_approx.npy')
    #     current_Q_dict_approx[LEFT] = np.load('Q_LEFT_approx.npy')
    #     current_Q_dict_approx[RIGHT] = np.load('Q_RIGHT_approx.npy')
    # else:
    #     #SAVE
    #     current_Q_dict_approx = computeQApproximation(average_prob, average_reward, reward_matrix, discount, N)
    #     np.save('Q_UP_approx.npy', current_Q_dict_approx[UP])
    #     np.save('Q_DOWN_approx.npy', current_Q_dict_approx[DOWN])
    #     np.save('Q_LEFT_approx.npy', current_Q_dict_approx[LEFT])
    #     np.save('Q_RIGHT_approx.npy', current_Q_dict_approx[RIGHT])

    # # Printing
    # print("UP\n", current_Q_dict_approx[UP])
    # print("DOWN\n", current_Q_dict_approx[DOWN])
    # print("LEFT\n", current_Q_dict_approx[LEFT])
    # print("RIGHT\n", current_Q_dict_approx[UP])
    
    # optimal_policy_approximation = getOptimalPolicy(current_Q_dict_approx)

