# Omid55
import numpy as np
import random as rand
import scipy.stats as stat
import math


class Problem:

    def __init__(self, npeople, ntasks, nmembers):
        self.npeople = npeople
        self.ntasks = ntasks
        self.nmembers = nmembers
        self.people = np.random.rand(npeople)
        self.tasks = np.random.rand(ntasks, nmembers)
        self.task_index = 0

    def set_people(self, people):
        self.people = people

    def set_tasks(self, tasks):
        self.tasks = tasks

    def pull_arm(self, team):
        skills = self.people[team]
        requirements = self.tasks[self.task_index]
        diff = np.setdiff1d(requirements, skills)
        if not diff:
            return 1
        else:
            return 0
        #return stat.binom.rvs(1, self.arms[selected_arm_index])


'''Different methods for choosing arm'''
# ground truth
def choose_optimal_arm(probl):
    for it in range(probl.ntasks):
        #arm = np.argmax(probl.arms)
        team = [0, 2]   #[2, 3]
        reward = probl.pull_arm(team)
        yield team, reward

# methods
def choose_arm_by_random(probl):
    for it in range(probl.ntasks):
        arm = rand.choice(range(probl.narms))
        reward = probl.pull_arm(arm)
        yield arm, reward

#def choose_arm_by_explore_few_then_exploit(bandit, iterations):  # Maximum Likelihood
#def choose_arm_by_ucb(bandit, iterations):
#def choose_arm_by_thompson_sampling(bandit, iterations):



'''Main function'''
def main():

    # params
    M = 2
    N = 5
    T = 1000

    probl = Problem(N, T, M)
    probl.set_people([1, 2, 5, 1, 3])
    probl.set_tasks(np.tile([1,5],(T,1)))
    print probl.tasks
    print probl.people

    # methods = [choose_optimal_arm, choose_arm_by_random]
    # for method in methods:
    #     rewards_mean = 0
    #     rewards_pow2_mean = 0
    #     c = 0
    #     for arm, reward in method(probl):
    #         rewards_mean = float(rewards_mean * c + reward) / (c+1)
    #         rewards_pow2_mean = float(rewards_pow2_mean * c + reward*reward) / (c+1)
    #         c += 1
    #
    #     stand_err_r = round(math.sqrt(rewards_pow2_mean - rewards_mean*rewards_mean)/math.sqrt(iterations),2)
    #     print method.__name__, ':\t', rewards_mean, '\t+/-\t', stand_err_r



if __name__ == "__main__":
    main()

