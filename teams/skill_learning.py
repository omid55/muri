# Omid55
"""Skill learning"""
"""Skill based task assignments=> Each person has a probability of success if being part of the
    team of k people solving one type of known task. This mean every person has a number between
    [0,1] (thus we have in fact a vector of n numbers falling in [0,1]) which is unkown for
    the algorithm. Oracle who knows all these information find the best team: (in tex format)
    \operatorname{arg\,max}_{team_j \in \forall teams} (1/k)*\sum_{i \in team_j}{P_i}
    Plus, we assume tasks are same type, for instance all are Sudoku with different numbers."""

import collections
import numpy as np
import random as rand
import scipy.stats as stat
import seaborn as sns
import pandas as pd
import itertools as itt
import math
import time
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sys


def pick_k_largest_indices(arr, k):
    large_indices = []
    for i in range(k):
        index = np.argmax(arr)
        large_indices.append(index)
        arr[index] = -sys.maxsize
    return large_indices


class Settings:
    def __init__(self, number_of_people, team_size, iterations, runs):
        self.number_of_people = number_of_people                        # Population size   (n)
        self.team_size = team_size                                      # Team size         (k)
        self.skill_probabilities = np.random.rand(number_of_people)     # unknown parameters
        self.iterations = iterations
        self.runs = runs
        self.total_possible_teams = []

    def get_total_possible_teams(self):
        if not self.total_possible_teams:
            self.total_possible_teams = list(itt.combinations(range(0, self.number_of_people), self.team_size))
            self.total_possible_teams = [tuple(sorted(team)) for team in self.total_possible_teams]    # make all teams as tuples
        return self.total_possible_teams

    def oracle_optimal_team(self):
        optimal_team = [i[0] for i in sorted(enumerate(self.skill_probabilities), key=lambda x:x[1], reverse=True)][:self.team_size]
        return optimal_team

    """team is a list of indices from self.skill_probabilities and task is
     always considered to be the same type for all teams.
     It returns status of the task at the end (success or failure)"""
    def assign_task(self, team):
        if not team or len(team) != self.team_size:
            raise ValueError('Size of the given team is different than settings team size.')
        # deterministic and only skill learning
        team_probability = np.mean(self.skill_probabilities[list(team)])
        # stochastic and only skill learning
        # deterministic and only edge learning
        # stochastic and only edge learning
        # deterministic and both learning
        # stochastic and both learning

        return stat.bernoulli.rvs(team_probability)


class Methods:
    def __init__(self, settings):
        self.settings = settings

    def get_selected_people_in_sample_format(self, team):
        selected_people = np.zeros(self.settings.number_of_people, dtype=int)
        selected_people[list(team)] = 1
        return selected_people

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def by_random(self):
        for iteration in range(self.settings.iterations):
            team = rand.sample(range(self.settings.number_of_people), self.settings.team_size)
            result = self.settings.assign_task(team)
            yield team, result

    def by_explore_then_exploit(self, epsilon = 0.2):    # explore_then_exploit
        exploration_count = epsilon * self.settings.iterations
        successes = collections.defaultdict(lambda: 0)    # map a team : the number of success in the past
        failures = collections.defaultdict(lambda: 0)     # map a team : the number of failure in the past
        explored_teams = []    # a list of teams that have been explored in the epslion part
        for iteration in range(self.settings.iterations):
            if iteration < exploration_count:
                team = tuple(sorted(rand.sample(range(self.settings.number_of_people), self.settings.team_size)))
            else:
                if not explored_teams:
                    explored_teams = list(set(list(successes.keys()) + list(failures.keys())))
                success_probability = [successes[a] / (successes[a] + failures[a]) for a in explored_teams]
                team = explored_teams[np.argmax(success_probability)]
            result = self.settings.assign_task(team)
            if result:
                successes[team] += 1
            else:
                failures[team] += 1
            yield team, result

    """Epsilon greedy is with probability of epsilon picks sample by random and by probability
        of 1 - epsilon it picks based on likelihood always"""
    def by_epsilon_greedy(self, epsilon = 0.2, dynamic=False):
        successes = collections.defaultdict(lambda: 0)  # map a team : the number of success in the past
        failures = collections.defaultdict(lambda: 0)  # map a team : the number of failure in the past
        for iteration in range(self.settings.iterations):
            if np.random.rand() <= epsilon or not len(successes) + len(failures):
                team = tuple(sorted((rand.sample(range(self.settings.number_of_people), self.settings.team_size))))
            else:
                explored_teams = list(set(list(successes.keys()) + list(failures.keys())))
                success_probability = [successes[a] / (successes[a] + failures[a]) for a in explored_teams]
                team = explored_teams[np.argmax(success_probability)]
            result = self.settings.assign_task(team)
            if result:
                successes[team] += 1
            else:
                failures[team] += 1
            yield team, result

    def by_likelihood(self):
        def get_success_chance(successes, failures, a):
            eps = 0.01
            exploration_eps = 0.2
            if not successes[a] + failures[a]:
                return exploration_eps
            else:
                return (eps + successes[a]) / (eps + (successes[a] + failures[a]))
        # starting the code for the main function
        successes = collections.defaultdict(lambda: 0)  # map a team : the number of success in the past
        failures = collections.defaultdict(lambda: 0)  # map a team : the number of failure in the past
        total_possible_teams = self.settings.get_total_possible_teams()
        for iteration in range(self.settings.iterations):
            success_probability = [get_success_chance(successes, failures, a) for a in total_possible_teams]
            np.divide(success_probability, sum(success_probability))
            team = total_possible_teams[np.argmax(self.softmax(success_probability))]
            result = self.settings.assign_task(team)
            if result:
                successes[team] += 1
            else:
                failures[team] += 1
            yield team, result

    def by_qlearning(self, epsilon=0.2, alpha=0.1, gamma=0.01):
        Q = collections.defaultdict(lambda: 0)   # map a team to value of Q in Q-learning
        total_possible_teams = self.settings.get_total_possible_teams()
        for iteration in range(self.settings.iterations):
            if np.random.rand() <= epsilon or not Q:
                team = tuple(sorted(rand.sample(range(self.settings.number_of_people), self.settings.team_size)))
            else:
                team = total_possible_teams[np.argmax(Q)]
            result = self.settings.assign_task(team)
            Q[team] += alpha * (result + gamma * max(Q.values()) - Q[team])
            yield team, result

    def by_ucb(self):
        all_possible_teams = self.settings.get_total_possible_teams()
        number_of_all_possible_teams = len(all_possible_teams)
        if self.settings.iterations <= number_of_all_possible_teams:
            raise ValueError('Number of iterations should be larger than number of teams.' + ' In fact, #iterations:' + str(
                self.settings.iterations) + ' and #teams:' + str(number_of_all_possible_teams))
        S = collections.defaultdict(lambda: 0)
        F = collections.defaultdict(lambda: 0)
        for team in all_possible_teams:
            reward = self.settings.assign_task(team)
            if reward:
                S[team] += 1
            else:
                F[team] += 1
            yield team, reward
        for it in range(self.settings.iterations - number_of_all_possible_teams):
            prob = [S[t] / float(S[t] + F[t]) + math.sqrt(
                2 * math.log(it + number_of_all_possible_teams) / float(S[t] + F[t])) for t in all_possible_teams]
            team = all_possible_teams[np.argmax(prob)]
            reward = self.settings.assign_task(team)
            team_str = str(team)[1:-1]
            if reward == 1:
                S[team_str] += 1
            else:
                F[team_str] += 1
            yield team, reward


    def by_thompson_sampling(self):
        all_possible_teams = self.settings.get_total_possible_teams()
        number_of_all_possible_teams = len(all_possible_teams)
        S = collections.defaultdict(lambda: 0)
        F = collections.defaultdict(lambda: 0)
        for it in range(self.settings.iterations):
            sampled_params = np.zeros(number_of_all_possible_teams)
            for t, team in enumerate(all_possible_teams):
                alpha = S[team] + 1
                beta = F[team] + 1
                sampled_params[t] = stat.beta.rvs(alpha, beta)
            team = all_possible_teams[np.argmax(sampled_params)]
            reward = self.settings.assign_task(team)
            if reward == 1:
                S[team] += 1
            else:
                F[team] += 1
            yield team, reward

    """by separate random variable for every person and apply an offline classification model to learn after
        som enough amount of iteration as the beginning dataset"""
    def by_offline_classification(self, epsilon = 0.2):
        training_samples_threshold = round(epsilon * self.settings.iterations)
        X = []
        y = []
        all_teams_in_format = []
        all_possible_teams = self.settings.get_total_possible_teams()
        for t in all_possible_teams:
            all_teams_in_format.append(self.get_selected_people_in_sample_format(t))
        classifier = LogisticRegression() #SVC()
        for iteration in range(self.settings.iterations):
            team = None
            if len(X) > training_samples_threshold:
                # train the whole classifier which is not reasonable and efficient approach
                classifier.fit(X, y)    # train from scratch
                # find one team that can be successful
                predicted_performance = classifier.predict(all_teams_in_format)
                if sum(predicted_performance) > 0:
                    team = all_possible_teams[np.argmax(predicted_performance)]
            # if the data is not enough for training or all teams were not selected as potential successful team
            if not team:
                team = tuple(sorted(rand.sample(range(self.settings.number_of_people), self.settings.team_size)))
            team_in_format = self.get_selected_people_in_sample_format(team)
            result = self.settings.assign_task(team)
            X.append(team_in_format)
            y.append(result)
            yield team, result

    """by separate random variable for every person and apply a classification model to learn after
        som enough amount of iteration as the beginning dataset"""
    def by_online_classification(self, epsilon = 0.2):
        training_samples_threshold = round(epsilon * self.settings.iterations)
        all_teams_in_format = []
        X = []
        y = []
        all_possible_teams = self.settings.get_total_possible_teams()
        for t in all_possible_teams:
            all_teams_in_format.append(self.get_selected_people_in_sample_format(t))
        classifier = SGDClassifier()
        for iteration in range(self.settings.iterations):
            team = None
            if iteration == training_samples_threshold:
                classifier.fit(X, y)
                del(X)
                del(y)
            if iteration > training_samples_threshold:
                # find one team that can be successful
                predicted_performance = classifier.predict(all_teams_in_format)
                if sum(predicted_performance) > 0:
                    team = all_possible_teams[np.argmax(predicted_performance)]
            # if the data is not enough for training or all teams were not selected as potential successful team
            if not team:
                team = tuple(sorted(rand.sample(range(self.settings.number_of_people), self.settings.team_size)))
            team_in_format = self.get_selected_people_in_sample_format(team)
            result = self.settings.assign_task(team)
            if iteration >= training_samples_threshold:
                classifier.partial_fit([team_in_format], [result])
            else:
                X.append(team_in_format)
                y.append(result)
            yield team, result

    '''Since nodes have separate probability and we try to find the maximum
        average; thus, we can pick node by node and that makes it so fast and efficient
        instead of C(N,M) possibilities we need N.'''
    def by_ucb_with_random_variable_per_node(self):
        # JUST FOR NOW   (JUST FOR INITIALIZATION OF UCB)   << CHECK HERE >>
        all_possible_teams = self.settings.get_total_possible_teams()
        number_of_all_possible_teams = len(all_possible_teams)
        if self.settings.iterations <= number_of_all_possible_teams:
            raise ValueError('Number of iterations should be larger than number of teams.' + ' In fact, #iterations:' + str(
            self.settings.iterations) + ' and #teams:' + str(number_of_all_possible_teams))
        # JUST FOR NOW   (JUST FOR INITIALIZATION OF UCB)   << CHECK HERE >>
        S = collections.defaultdict(lambda: 0)
        F = collections.defaultdict(lambda: 0)
        for team in all_possible_teams:
            reward = self.settings.assign_task(team)
            for i in team:
                if reward:
                    S[i] += 1
                else:
                    F[i] += 1
            yield team, reward
        for it in range(self.settings.iterations - number_of_all_possible_teams):
            prob = np.zeros(self.settings.number_of_people)
            for i in range(self.settings.number_of_people):
                ni = float(S[i] + F[i])
                prob[i] += S[i] / ni + math.sqrt(2 * math.log(it + number_of_all_possible_teams) / ni)
            ##team = tuple(sorted(np.argsort(prob)[-self.settings.team_size:]))
            team = tuple(sorted(pick_k_largest_indices(prob, self.settings.team_size)))
            reward = self.settings.assign_task(team)
            for i in team:
                if reward:
                    S[i] += 1
                else:
                    F[i] += 1
            yield team, reward

    def by_thompson_sampling_with_random_variable_per_node(self):
        S = collections.defaultdict(lambda: 0)
        F = collections.defaultdict(lambda: 0)
        for it in range(self.settings.iterations):
            sampled_params = np.zeros(self.settings.number_of_people)
            for i in range(self.settings.number_of_people):
                alpha = S[i] + 1
                beta = F[i] + 1
                sampled_params[i] = stat.beta.rvs(alpha, beta)
            ##team = tuple(sorted(np.argsort(sampled_params)[-self.settings.team_size:]))
            team = tuple(sorted(pick_k_largest_indices(sampled_params, self.settings.team_size)))
            reward = self.settings.assign_task(team)
            for i in team:
                if reward:
                    S[i] += 1
                else:
                    F[i] += 1
            yield team, reward

    """run all of methods and computing their status on the task"""
    def solve(self):
        iterations = self.settings.iterations
        runs = self.settings.runs
        # figure parameters
        with_start_from_zero = False
        figure_points = 20
        sns.set(rc={"figure.figsize": (10, 10)})
        sns.set_palette("Set1", 8)
        all_methods = [self.by_random, self.by_explore_then_exploit,
                       #self.by_qlearning, self.by_epsilon_greedy, self.by_likelihood,
                       #self.by_ucb, self.by_thompson_sampling,
                       self.by_online_classification, self.by_offline_classification,
                       #self.by_ucb_with_random_variable_per_node,
                       self.by_thompson_sampling_with_random_variable_per_node]
        success_for_runs = pd.DataFrame(columns=['Point', 'Method', 'Run', 'Success Ratio'])
        regret_for_runs = pd.DataFrame(columns=['Point', 'Method', 'Run', 'Regret'])
        optimal_team = self.settings.oracle_optimal_team()
        counter = 0
        for method_index, method in enumerate(all_methods):
            print('\nMethod ' + method.__name__ + ': ', end=" ")
            start_time = time.time()
            for run in range(runs):
                print('*', end="")
                regret = win_no = task_no = 0
                if with_start_from_zero:
                    success_for_runs.loc[counter] = [0, method.__name__, run, 0]
                    regret_for_runs.loc[counter] = [0, method.__name__, run, 1]
                    counter += 1
                for running_iteration, (team, success) in enumerate(method()):
                    # computing win ratio
                    if success:
                        win_no += 1
                    task_no += 1
                    # computing regret
                    optimal_success = self.settings.assign_task(optimal_team)
                    regret += optimal_success - success
                    if not (running_iteration+1) % (iterations / figure_points):
                        success_for_runs.loc[counter] = [task_no, method.__name__, run, win_no / task_no]
                        regret_for_runs.loc[counter] = [task_no, method.__name__, run, regret / (running_iteration + 1)]
                        counter += 1
            print(' (in ', round(time.time() - start_time,2), ' seconds)', end="")
        sns.tsplot(time='Point', value='Success Ratio', unit='Run', condition='Method', data=success_for_runs)
        sns.plt.show()
        sns.tsplot(time='Point', value='Regret', unit='Run', condition='Method', data=regret_for_runs)
        sns.plt.show()


def run(number_of_people=10, team_size=2, iterations=1000, runs=10):
    setting = Settings(number_of_people, team_size, iterations, runs)
    methods = Methods(setting)
    methods.solve()


def main():
    run()


if __name__ == '__main__':
    main()
