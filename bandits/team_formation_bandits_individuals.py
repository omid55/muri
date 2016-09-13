# Omid55
import numpy as np
import random as rand
import scipy.stats as stat
import math
import itertools as itt
import collections
import matplotlib.pyplot as plt
import time
import seaborn as sns
% matplotlib
inline

'''Problem class'''
'''Each person has a probability of success in
 the task and network and weight between does not matter'''


class Problem:
    # npeople
    # probabilities: [npeople,1]
    # team_size
    # optimal_team

    def __init__(self, npeople, team_size):
        self.npeople = npeople
        self.probabilities = np.random.rand(npeople)
        self.team_size = team_size
        self.compute_optimal_team()
        self.STOCHASTICITY = 0.05

    def set_npeople(self, npeople):
        self.npeople = npeople
        self.probabilities = np.random.rand(npeople)
        self.compute_optimal_team()

    def set_team_size(self, team_size):
        self.team_size = team_size
        self.compute_optimal_team()

    def set_STOCHASTICITY(self, STOCHASTICITY):
        self.STOCHASTICITY = STOCHASTICITY

    def set_probabilities(self, probabilities):
        self.probabilities = np.array(probabilities)
        self.compute_optimal_team()

    def compute_optimal_team(self):
        v = sorted(self.probabilities, reverse=True)[self.team_size - 1]
        self.optimal_team = np.where(self.probabilities >= v)[0]

    def pull_arm(self, team):  # objective (fitness) function
        assert len(team) == self.team_size, 'team size should not be different than ' + self.team_size + ' .'
        prob = (1 - self.STOCHASTICITY) * self.compute_value_of_team(team) / self.compute_value_of_team(
            self.optimal_team)
        return stat.bernoulli.rvs(prob)  # stat.binom.rvs(1, prob)

    # the average of probability of team members
    def compute_value_of_team(self, team):
        v = 0
        for person in team:
            v += self.probabilities[person] / float(len(team))
        return v


'''Different methods for choosing team'''


# ground truth
def choose_optimal_team(problem, iterations):
    for it in range(iterations):
        reward = problem.pull_arm(problem.optimal_team)
        yield problem.optimal_team, reward


# methods
def choose_team_by_random(problem, iterations):
    for it in range(iterations):
        team = rand.sample(range(problem.npeople), problem.team_size)
        reward = problem.pull_arm(team)
        yield team, reward


def choose_team_by_explore_few_then_exploit(problem, iterations):  # Maximum Likelihood
    exploration_percent = 10
    exploration_count = iterations * exploration_percent / 100.0
    teams = []
    S = collections.defaultdict(lambda: 0)
    F = collections.defaultdict(lambda: 0)
    for it in range(iterations):
        if it < exploration_count:
            team = sorted(rand.sample(range(problem.npeople), problem.team_size))
            reward = problem.pull_arm(team)
            if reward == 1:
                S[str(team)[1:-1]] += 1
            else:
                F[str(team)[1:-1]] += 1
            yield team, reward
        else:
            if not teams:
                teams = list(set(list(S.keys()) + list(F.keys())))
            prob = [S[a] / (S[a] + F[a]) for a in teams]
            team_str = teams[np.argmax(prob)]
            team = list(map(int, team_str.split(',')))
            reward = problem.pull_arm(team)
            if reward == 1:
                S[team_str] += 1
            else:
                F[team_str] += 1
            yield team, reward


'''Main function'''


def main():
    auto = 0
    figure_points = 20

    if auto:
        # params
        number_of_people = 10
        team_size = 4
        iterations = 5000
        probl = Problem(number_of_people, team_size)
    else:
        number_of_people = 4
        team_size = 2
        iterations = 1000
        probl = Problem(number_of_people, team_size)
        probl.set_probabilities([0.9, 0.1, 0.4, 0.7])
        print('Unkown probabilities:', probl.probabilities)
    print('Optimal team:', probl.optimal_team, '\n\n')

    sns.set(rc={"figure.figsize": (10, 10)})
    sns.set_palette("Set1", 8)
    po = np.arange(iterations / figure_points, iterations + iterations / figure_points, iterations / figure_points)
    methods = [choose_optimal_team, choose_team_by_random, choose_team_by_explore_few_then_exploit]
    for method in methods:
        start = time.time()
        rewards_mean = 0
        rewards_pow2_mean = 0
        success = np.zeros(figure_points)
        cnt = 0.0
        win = 0.0
        for team, reward in method(probl, iterations):
            rewards_mean = float(rewards_mean * cnt + reward) / (cnt + 1)
            rewards_pow2_mean = float(rewards_pow2_mean * cnt + reward * reward) / (cnt + 1)
            if reward:
                win += 1.0
            cnt += 1.0
            if not cnt % (iterations / figure_points):
                success[int(cnt / (iterations / figure_points)) - 1] = win / cnt

        plt.plot(po, success)
        rewards_std = round(math.sqrt(rewards_pow2_mean - rewards_mean * rewards_mean) / math.sqrt(iterations), 2)
        print(
        method.__name__, '=>\tlast team: ', team, '=', probl.compute_value_of_team(team), ',\trewards: ', rewards_mean,
        '+/-', rewards_std, ', in ', round(time.time() - start, 2), ' seconds.')

    plt.xlabel('# Iteration')
    plt.ylabel('Success ratio until this iteration')
    plt.legend(['optimal', 'random', 'explore then exploit', 'UCB', 'Thompson', 'UCB for each', 'Thompson for each',
                'Thompson both'], loc='best')
    plt.show()


if __name__ == "__main__":
    main()
