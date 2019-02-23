import math
import numpy as np
import random
import os.path
from matplotlib import pyplot as plt




class Domain:

    def __init__(self, reward_matrix, discount, actions, beta=0):
        self.discount = discount
        self.reward_matrix = reward_matrix
        self.dim_x =  reward_matrix.shape[0]
        self.dim_y =  reward_matrix.shape[1]
        self.actions = actions
        self.beta = beta
        
    
    def _nextState(self, state, action):
        """
        Returns the next state given the current state
        and the action to perform.
        Arguments:
        ----------
        - state : tuple giving current state coordinates
        - action: tuple giving the action to take
        Returns:
        --------
        Tuple corresponding to the coordinates of the next state
        """

        maximum_x = max(state[0] + action[0], 0)
        maximum_y = max(state[1] + action[1], 0)
        return (min(maximum_x, self.dim_x - 1), min(maximum_y, self.dim_y - 1))

    def dynamics(self, state, action):
        """
        Returns the next state given the current state
        and the action to perform.
        If the domain is stochastic, the returned state may be
        (0,0) regardless of the state given, depending on the 
        parameter `self.beta`.
        Arguments:
        ----------
        - state : tuple giving current state coordinates
        - action: tuple giving the action to take
        Returns:
        --------
        Tuple corresponding to the coordinates of the next state
        """

        w = random.random()

        if w <= 1 - self.beta:
            return self._nextState(state, action)
        else:
            return (0,0)

    
    def rewardAtPosition(self, coords):
        """Returns the reward from a given cell in the 
        domains `reward_matrix` whose coordinates are given by a tuple
        Arguments:
        ----------
        - coords : tuple describing the cell from which to extract the reward

        Returns:
        ----------
        - integer reward
        """
        return self.reward_matrix[coords]


    def expectedReward(self, state, action):
        """
        Computes the expected reward of a state
        Uses the `self.beta` domain variable
        Arguments:
        ---------
        - state: Tuple containing the coordinates of the current state
        - action: Tuple containing the action to take in the current state 

        Returns:
        --------
        Integer value fo the expected reward of the cell
        """

        deterministic = self.rewardAtPosition(self._nextState(state, action))
        stochastic = self.rewardAtPosition((0, 0))
        return (1 - self.beta) * deterministic + self.beta * stochastic

    def probability(self, state, action, test_state):
        """
        Computes the probability of reaching a next state
        by taking the given action in the current state. 
        Arguments:
        ---------
        - state: Tuple containing the coordinates of the current state
        - action: Tuple containing the action to take in the current state 
        - test_state: Tuple containing the coordiantes of the next state to test
        Returns:
        --------
        Float value between 0 and 1
        """

        next_state = self._nextState(state, action)

        if self.beta == 0: # Deterministic
            if next_state == test_state:
                return 1 
            else:
                return 0
        else: # Stochastic
            if test_state == (0,0):
                return  self.beta
            elif next_state == test_state:
                return (1 - self.beta)  
            else:
                return 0


class Agent:

    def __init__(self, domain):
        self.domain = domain
        self.Q = None
        self.J = None


    def computeJ(self, policy, N):
        """
        Computes the cumulative expected reward for a whole grid
        Arguments:
        ----------
        - policy: dictionary containing the action to take at each state
        - N : number of iterations to perform
        Returns:
        ----------
        - Matrix with the same shape as reward_matrix containing
        in each cell the cumulative expected reward of that state 
        """

        current_J = np.zeros_like(self.domain.reward_matrix, dtype=float)
        for n in range(N):
            previous_J = current_J

            for x in range(self.domain.dim_x):
                for y in range(self.domain.dim_y):
                    next_point = self.domain.dynamics((x, y), policy[(x,y)])

                    current_J[(x, y)] = self.domain.expectedReward((x,y), policy[(x,y)]) \
                        + self.domain.discount * \
                        ((1 - self.domain.beta) *
                        previous_J[next_point] + self.domain.beta * previous_J[(0, 0)])
        self.J = current_J
        return current_J
    

    def computeProba(self, action):
        """
        Compute the probability to reach any next state from any state
        using the given action

        Arguments:
        ----------
        - action: tuple giving the next action to take

        Returns:
        ----------
        - Dictionnary with current_state as key and probability grids
        giving the probability of reaching any next state 
        """

        # Dictionnary storing probability grids for any current_state, next_state pair
        # after having taken action `action`
        # Contains dim_x * dim_y entries of (dim_x, dim_y) dimension matrices
        # Key is current state
        prob_for_next_states = {}
        for x in range(self.domain.dim_x):
            for y in range(self.domain.dim_y):
                prob = np.zeros_like(reward_matrix, dtype=float)
                for next_x in range(self.domain.dim_x):
                    for next_y in range(self.domain.dim_y):
                        prob[(next_x, next_y)] = self.domain.probability((x,y), action, (next_x, next_y))
                prob_for_next_states[(x,y)] = prob

        return prob_for_next_states


    def computeQ(self, N):
        """
        Computes the cumulative expected reward for state action pairs

        Arguments:
        ----------
        - N : number of iterations to perform

        Returns:
        ----------
        - Dictionnary containing entries for each different action
        Keys are actions and values are cumulative expected reward matrices
        of the domain
        """

        current_Q_dict = {}
        for u in self.domain.actions:
            current_Q_dict[u] = np.zeros_like(self.domain.reward_matrix, dtype=float)

        # Compute the whole probability grid for every current_state , next_state pair
        # for a given action
        for n in range(N):
            print("N = {}".format(n))
            previous_Q_dict = current_Q_dict

            for x in range(self.domain.dim_x):
                for y in range(self.domain.dim_y):
                    for u in self.domain.actions:
                        prob_dict = self.computeProba(u)
                        expectation = 0

                        # Compute the expectation of the reward of previous states
                        for next_x in range(self.domain.dim_x):
                            for next_y in range(self.domain.dim_y):
                                next_state = (next_x, next_y)

                                # Finding maximum cumulative reward of the previous iteration
                                max_reward = -math.inf
                                for u_prime in self.domain.actions:
                                    current_Q = current_Q_dict[u_prime]
                                    if current_Q[next_state] > max_reward:
                                        max_reward = previous_Q_dict[u_prime][next_state]

                                expectation += prob_dict[(x,y)][next_state] * max_reward    

                        reward = self.domain.expectedReward((x,y), u)
                        current_Q_dict[u][(x,y)] = reward + self.domain.discount * expectation
        self.Q = current_Q_dict
        return current_Q_dict
    
    def computePolicy(self):
        """
        Computes the policy from the state action value function

        Arguments: /
        ----------
        Returns:
        --------
        Dictionary with states as keys and actions as values
        """

        if(not self.Q):
            self.computeQ(1000)

        policy = {}
        # Policy from the reccurence equation Q
        for x in range(self.domain.dim_x):
            for y in range(self.domain.dim_y):
                best_action = None
                max_reward = -math.inf
                for u in self.domain.actions:
                    reward = self.Q[u][(x,y)]
                    if(reward > max_reward):
                        max_reward = reward
                        best_action = u
                policy[(x,y)] = best_action
        # # Printing
        # for i in policy:
        #     if policy[i] == UP:
        #         print("{} : UP".format(i))
        #     elif policy[i] == DOWN:
        #         print("{} : DOWN".format(i))
        #     elif policy[i] == LEFT:
        #         print("{} : LEFT".format(i))
        #     elif policy[i] == RIGHT:
        #         print("{} : RIGHT".format(i))
        return policy

    def computeApproximatePolicy(self, traj_length, nb_traj, N):
        """
        Computes the policy from the approximate state action value function
        which is computed using trajectories

        Arguments:
        ----------
        - traj_length: length of the trajetories
        - nb_traj: number of trajectories 
        - N: number of iterations to perform

        Returns:
        --------
        - Dictionary with states as keys and actions as values
        """

        Q_approx = agent_det.computeQApproximation(traj_length, nb_traj, N)

        policy = {}
        # Policy from the reccurence equation Q
        for x in range(self.domain.dim_x):
            for y in range(self.domain.dim_y):
                best_action = None
                max_reward = -math.inf
                for u in self.domain.actions:
                    reward = Q_approx[u][(x,y)]
                    if(reward > max_reward):
                        max_reward = reward
                        best_action = u
                policy[(x,y)] = best_action
        # # Printing
        # for i in policy:
        #     if policy[i] == UP:
        #         print("{} : UP".format(i))
        #     elif policy[i] == DOWN:
        #         print("{} : DOWN".format(i))
        #     elif policy[i] == LEFT:
        #         print("{} : LEFT".format(i))
        #     elif policy[i] == RIGHT:
        #         print("{} : RIGHT".format(i))
        return policy


    def createTrajectory(self, N):
        """
        Creates a trajectory h in the reward_matrix grid
        that is `N` states long, with random ACTIONS and a
        random starting state from the domain.
        h is of the form (state_0, action0, reward_0, ..., state_N)

        Arguments:
        ----------
        - N : length of the trajectory

        Returns:
        ----------
        - List containing the trajectory
        """
        # Creating trajectory h from a list of ACTIONS and a starting state
        h_start = (random.randint(0,4), random.randint(0,4))
        h_ACTIONS = [random.choice(self.domain.actions) for i in range(N)]

        h = list()
        h.append(h_start)

        # Successive iterations
        for u in h_ACTIONS:
            next_state = self.domain.dynamics(h[-1], u)
            h.append(u)
            h.append(self.domain.rewardAtPosition(next_state))
            h.append(next_state)
        return h


    def _computeAverageReward(self, traj_length, nb_traj, plot_convergence=False):
        """
        Computes a good approximation of the reward r(x,u) from the trajectory h
        h is of the form (state_0, action0, reward_0, ..., state_N)

        Arguments:
        ----------
        - traj_length: length of the trajetories
        - nb_traj: number of trajectories 
        - plot_convergence: boolean value specifying if the user wants to compute
        and plot the convergence of the average reward to the true reward

        Returns:
        ----------
        - List containing the trajectory
        """

        nb_occurences = {}
        reward_sum = {}
        average_reward = {}
        convergence_diff = []
        convergence_nbpoints = 0
        all_state_diff = np.zeros_like(reward_matrix, dtype=float)

        # Init
        for u in self.domain.actions:
            nb_occurences[u] = np.zeros_like(reward_matrix, dtype=int)
            reward_sum[u] = np.zeros_like(reward_matrix, dtype=int)
            average_reward[u] = np.zeros_like(reward_matrix, dtype=int)

        # Create 1000 trajectories and compute the average reward
        # of each state
        for t in range(nb_traj):
            traj = self.createTrajectory(traj_length)
            i = 0
            while i < traj_length -1 :
                nb_occurences[traj[i+1]][traj[i]] += 1
                reward_sum[traj[i+1]][traj[i]] += traj[i + 2]  
                i = i + 3

            # Compute convergence
            if plot_convergence:
                convergence_nbpoints += 1

                action = self.domain.actions[0]
                for x in range(self.domain.dim_x):
                    for y in range(self.domain.dim_y):
                        approx_reward = np.nan_to_num(np.divide(reward_sum[action], nb_occurences[action]), False)
                        true_reward = self.domain.expectedReward((x,y), action)
                        all_state_diff = abs(approx_reward - true_reward)
                convergence_diff.append(np.amax(all_state_diff))

                plt.plot(range(convergence_nbpoints), convergence_diff)
                plt.show()

        np.seterr(divide='ignore')
        for u in self.domain.actions:
            average_reward[u] = np.nan_to_num(np.divide(reward_sum[u], nb_occurences[u]), False)
        return average_reward


    def _computeAverageProbability(self, traj_length, nb_traj,plot_convergence=False):
        """
        Computes a good approximation of the probability p(x'| x,u) from the trajectory h
        h is of the form (state_0, action0, reward_0, ..., state_N)

        Arguments:
        ----------
        - traj_length: length of the trajetories
        - nb_traj: number of trajectories 
        - plot_convergence: boolean value specifying if the user wants to compute
        and plot the convergence of the average reward to the true reward
        Returns:
        ----------
        - List containing the trajectory
        """
        nb_times_from_curr_to_next = {}
        nb_times_through_curr = {}
        average_prob = {}

        # Init
        for x in range(self.domain.dim_x):
            for y in range(self.domain.dim_y):
                for u in self.domain.actions:
                    nb_times_from_curr_to_next[((x,y),u)] = np.zeros_like(self.domain.reward_matrix, dtype=int) 
                    nb_times_through_curr[((x,y),u)] = 0
                    average_prob[((x,y),u)] = np.zeros_like(self.domain.reward_matrix, dtype=int)

        # Create 1000 trajectories and compute the number of times we pass through 
        # different states and where we came from to reach those
        for nb_traj in range(nb_traj):
            traj = self.createTrajectory(traj_length)
            i = 0
            while i < traj_length - 1:
                nb_times_from_curr_to_next[(traj[i], traj[i+1])][traj[i + 3]] += 1
                nb_times_through_curr[(traj[i], traj[i+1])] += 1
                i = i + 3

        # Compute average probability for each state
        for x in range(self.domain.dim_x):
            for y in range(self.domain.dim_y):
                for u in self.domain.actions:
                    np.seterr(divide='ignore')
                    average_prob[((x,y),u)] = np.nan_to_num(np.divide(nb_times_from_curr_to_next[((x,y),u)], 
                                                                    nb_times_through_curr[((x,y),u)]), False)
        return average_prob


    def computeQApproximation(self, traj_length, nb_traj, N):
        """
        Computes an approximation of the cumulative expected reward 
        for state action pairs using trajectories
        
        Arguments:
        ----------
        - traj_length: length of the trajetories
        - nb_traj: number of trajectories
        - N : number of iterations to perform

        Returns:
        ----------
        - Dictionnary containing entries for each different action.
        Keys are actions and values are cumulative expected reward matrices
        with the same shape as `reward_matrix`.
        """

        current_Q_dict = {}
        for u in self.domain.actions:
            current_Q_dict[u] = np.zeros_like(self.domain.reward_matrix, dtype=float)

        average_prob = self._computeAverageProbability(traj_length, nb_traj)
        average_reward = self._computeAverageReward(traj_length, nb_traj)


        # Compute the whole probability grid for every current_state , next_state pair
        # for a given action
        for n in range(N):
            print("N = {}".format(n))
            previous_Q_dict = current_Q_dict

            for x in range(self.domain.dim_x):
                for y in range(self.domain.dim_y):
                    for u in self.domain.actions:
                        Q_sum = 0

                        # Compute the sum over all states of the reward of previous states
                        for next_x in range(self.domain.dim_x):
                            for next_y in range(self.domain.dim_y):
                                next_state = (next_x, next_y)

                                # Finding maximum cumulative reward of the previous iteration
                                # and multiplying bu the probability of landing
                                # in that state
                                if(average_prob[((x,y),u)][next_state] != 0.0):
                                    max_reward = -math.inf
                                    for u_prime in self.domain.actions:
                                        current_Q = current_Q_dict[u_prime]
                                        if current_Q[next_state] > max_reward:
                                            max_reward = previous_Q_dict[u_prime][next_state]

                                    Q_sum += average_prob[((x,y),u)][next_state] * max_reward

                        reward = average_reward[u][(x,y)]
                        current_Q_dict[u][(x,y)] = reward + self.domain.discount * Q_sum
        return current_Q_dict




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

    # CONSTANTS
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    ACTIONS = [UP, DOWN, LEFT, RIGHT]

    det_domain = Domain(reward_matrix, 0.99, ACTIONS, 0)
    stoch_domain = Domain(reward_matrix, 0.99, ACTIONS, 0.2)
    agent_det = Agent(det_domain) 
    agent_stoch = Agent(stoch_domain) 

    N_opt = 1000
    N_approx = 5000

    """======== Expected return of a policy - QUESTION 3 ==================="""

    policy = {}
    for x in range(reward_matrix.shape[0]):
        for y in range(reward_matrix.shape[1]):
            policy[(x,y)] = RIGHT

    print("Initial matrix\n", reward_matrix)
    print("Cumulative reward with deterministic dynamics, 10000 iterations :\n",
    agent_det.computeJ(policy, 10000))
    print("Cumulative expected reward with stochastic dynamics, beta = 0.5, 10000 iterations :\n",
    agent_stoch.computeJ(policy, 10000))



    """===================== Optimal policy - QUESTION 4 ==================="""

    # Initialization
    optimal_policy = agent_det.computePolicy()
    print("QUESTION 4: Value function\n", agent_det.computeJ(optimal_policy, 10000))


    """===================== System identification - QUESTION 5 ============"""

    optimal_policy_approximation = agent_det.computeApproximatePolicy(1000, 1000, N_approx)
    print("QUESTION 5: Value function approximation\n",agent_det.computeJ(optimal_policy_approximation, 10000))


