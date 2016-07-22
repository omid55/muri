# Omid55
import numpy as np
import random as rand
#from scipy.stats import binom
import scipy.stats as stat
import math

'''Bandit class'''
class Bandit:
    #narms
    #arms

    def __init__(self, narms):
        self.narms = narms
        self.arms = np.random.rand(narms)

    def set_arms(self, arms):
        self.arms = arms

    def pull_arm(self, selected_arm_index):
        return stat.binom.rvs(1, self.arms[selected_arm_index])


'''Different methods for choosing arm'''
# ground truth
def choose_optimal_arm(bandit, iterations):
    for it in range(iterations):
        arm = np.argmax(bandit.arms)
        reward = bandit.pull_arm(arm)
        yield arm, reward

# methods
def choose_arm_by_random(bandit, iterations):
    for it in range(iterations):
        arm = rand.choice(range(bandit.narms))
        reward = bandit.pull_arm(arm)
        yield arm, reward

def choose_arm_by_explore_few_then_exploit(bandit, iterations):  # Maximum Likelihood
    exploration_count = 100
    S = np.zeros(bandit.narms)
    F = np.zeros(bandit.narms)
    for it in range(iterations):
        if it < exploration_count:
            arm = rand.choice(range(bandit.narms))
            reward = bandit.pull_arm(arm)
            if reward == 1:
                S[arm] += 1
            else:
                F[arm] += 1
            yield arm, reward
        else:
            prob = [S[a]/(S[a]+F[a]) for a in range(bandit.narms)]
            arm = np.argmax(prob)
            reward = bandit.pull_arm(arm)
            if reward == 1:
                S[arm] += 1
            else:
                F[arm] += 1
            yield arm, reward


def choose_arm_by_ucb(bandit, iterations):
    assert iterations>bandit.narms, 'Number of iterations should be larger than number of arms.'
    S = np.zeros(bandit.narms)
    F = np.zeros(bandit.narms)
    for a in range(bandit.narms):
        reward = bandit.pull_arm(a)
        if reward == 1:
            S[a] += 1
        else:
            F[a] += 1
        yield a, reward
    for it in range(iterations-bandit.narms):
        prob = [S[a]/(S[a]+F[a]) + math.sqrt(2*math.log(it+bandit.narms)/(S[a]+F[a])) for a in range(bandit.narms)]
        arm = np.argmax(prob)
        reward = bandit.pull_arm(arm)
        if reward == 1:
            S[arm] += 1
        else:
            F[arm] += 1
        yield arm, reward

def choose_arm_by_thompson_sampling(bandit, iterations):
    S = np.zeros(bandit.narms)
    F = np.zeros(bandit.narms)
    for it in range(iterations):
        sampled_params = np.zeros(bandit.narms)
        for a in range(bandit.narms):
            alpha = S[a] + 1
            beta = F[a] + 1
            sampled_params[a] = stat.beta.rvs(alpha, beta)
        arm = np.argmax(sampled_params)
        reward = bandit.pull_arm(arm)
        if reward == 1:
            S[arm] += 1
        else:
            F[arm] += 1
        yield arm, reward



'''Main function'''
def main():

    # params
    number_of_arms = 5
    iterations = 1000

    bandit = Bandit(number_of_arms)
    bandit.set_arms([0.2, 0.3, 0.5, 0.9, 0.1])  # comment it out to see diverse problems << CHECK HERE >>
    print bandit.arms

    methods = [choose_optimal_arm, choose_arm_by_random, choose_arm_by_explore_few_then_exploit, choose_arm_by_ucb, choose_arm_by_thompson_sampling]
    for method in methods:
        rewards_mean = 0
        rewards_pow2_mean = 0
        c = 0
        for arm, reward in method(bandit, iterations):
            rewards_mean = float(rewards_mean * c + reward) / (c+1)
            rewards_pow2_mean = float(rewards_pow2_mean * c + reward*reward) / (c+1)
            c += 1

        stand_err_r = round(math.sqrt(rewards_pow2_mean - rewards_mean*rewards_mean)/math.sqrt(iterations),2)
        print method.__name__, ':\t', rewards_mean, '\t+/-\t', stand_err_r


if __name__ == "__main__":
    main()

