# Omid55
import numpy as np
import random as rand
import scipy.stats as stat
import math
import itertools as itt

'''prob class'''
class Problem:
    # npeople
    # weights
    # team_size
    # optimal_team

    # << CHECK HERE >>  WEIGHTS SHOULD BE SYMMETRY

    def __init__(self, npeople, team_size):
        self.npeople = npeople
        self.team_size = team_size
        self.weights = np.random.rand(npeople, npeople)
        # << CHECK HERE >>  WEIGHTS SHOULD BE SYMMETRY
        # << CHECK HERE >>  ALSO REMOVE DIAG OF WEIGHTS
        # self.weights = np.max(self.weights, self.weights.transpose())
        self.optimal_team = []
        self.STOCHASTICITY = 0.05

    def set_weights(self, weights):
        self.weights = weights

    def pull_team(self, team):    # objective (fitness) function
        assert len(team) == self.team_size, 'team size should not be different than ' + self.team_size + ' .'
        prob = (1-self.STOCHASTICITY) * self.compute_value_of_team(team) / self.compute_value_of_team(self.optimal_team)
        return stat.binom.rvs(1, prob)

    def compute_value_of_team(self, team):
        w = 0
        for i in range(0, len(team)-1):
            for j in range(i+1, len(team)):
                w += self.weights[team[i]][team[j]]
        return w


'''Different methods for choosing team'''
# ground truth
def choose_optimal_team(problem, iterations):
    for it in range(iterations):
        if not problem.optimal_team:
            all_teams = itt.combinations(range(0, problem.npeople), problem.team_size)
            max_v = -999
            for team in all_teams:
                v = problem.compute_value_of_team(team)
                #print(str(team) + ': ' + str(v))
                if v > max_v:
                    problem.optimal_team = team
                    max_v = v
            #print(problem.optimal_team)
        reward = problem.pull_team(problem.optimal_team)
        yield team, reward

# methods
def choose_team_by_random(problem, iterations):
    for it in range(iterations):
        team = rand.sample(range(problem.npeople), problem.team_size)
        reward = problem.pull_team(team)
        yield team, reward

# def choose_team_by_explore_few_then_exploit(prob, iterations):  # Maximum Likelihood
#     exploration_count = 100
#     S = np.zeros(prob.nteams)
#     F = np.zeros(prob.nteams)
#     for it in range(iterations):
#         if it < exploration_count:
#             team = rand.choice(range(prob.nteams))
#             reward = problem.pull_team(team)
#             if reward == 1:
#                 S[team] += 1
#             else:
#                 F[team] += 1
#             yield team, reward
#         else:
#             prob = [S[a]/(S[a]+F[a]) for a in range(prob.nteams)]
#             team = np.argmax(prob)
#             reward = problem.pull_team(team)
#             if reward == 1:
#                 S[team] += 1
#             else:
#                 F[team] += 1
#             yield team, reward
#
#
# def choose_team_by_ucb(prob, iterations):
#     assert iterations>prob.nteams, 'Number of iterations should be larger than number of teams.'
#     S = np.zeros(prob.nteams)
#     F = np.zeros(prob.nteams)
#     for a in range(prob.nteams):
#         reward = problem.pull_team(a)
#         if reward == 1:
#             S[a] += 1
#         else:
#             F[a] += 1
#         yield a, reward
#     for it in range(iterations-prob.nteams):
#         prob = [S[a]/(S[a]+F[a]) + math.sqrt(2*math.log(it+prob.nteams)/(S[a]+F[a])) for a in range(prob.nteams)]
#         team = np.argmax(prob)
#         reward = problem.pull_team(team)
#         if reward == 1:
#             S[team] += 1
#         else:
#             F[team] += 1
#         yield team, reward
#
# def choose_team_by_thompson_sampling(prob, iterations):
#     S = np.zeros(prob.nteams)
#     F = np.zeros(prob.nteams)
#     for it in range(iterations):
#         sampled_params = np.zeros(prob.nteams)
#         for a in range(prob.nteams):
#             alpha = S[a] + 1
#             beta = F[a] + 1
#             sampled_params[a] = stat.beta.rvs(alpha, beta)
#         team = np.argmax(sampled_params)
#         reward = problem.pull_team(team)
#         if reward == 1:
#             S[team] += 1
#         else:
#             F[team] += 1
#         yield team, reward
#


'''Main function'''
def main():

    # params
    number_of_people = 5
    team_size = 3
    iterations = 1000

    prob = Problem(number_of_people, team_size)
    prob.set_weights([[0,0.75,0.06,0.3,0.5], [0.75,0,0.88,0.46,0.95], [0.06,0.88,0,0.7,0.9], [0.3,0.46,0.7,0,0.02], [0.5,0.95,0.9,0.02,0]])  # comment it out to see diverse problems << CHECK HERE >>
    print prob.weights, '\n\n'

    methods = [choose_optimal_team, choose_team_by_random]   #, choose_team_by_explore_few_then_exploit, choose_team_by_ucb, choose_team_by_thompson_sampling]
    for method in methods:
        rewards_mean = 0
        rewards_pow2_mean = 0
        c = 0
        for team, reward in method(prob, iterations):
            rewards_mean = float(rewards_mean * c + reward) / (c+1)
            rewards_pow2_mean = float(rewards_pow2_mean * c + reward*reward) / (c+1)
            c += 1

        stand_err_r = round(math.sqrt(rewards_pow2_mean - rewards_mean*rewards_mean)/math.sqrt(iterations),2)
        print method.__name__, ':\t', rewards_mean, '\t+/-\t', stand_err_r


if __name__ == "__main__":
    main()

