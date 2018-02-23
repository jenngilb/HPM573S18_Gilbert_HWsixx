import numpy as np

import scr.SamplePathClass as SamplePathSupport
import scr.FigureSupport as Fig


class Game(object):
    def __init__(self, id, prob_head):
        self._id = id
        self._rnd = np.random
        self._rnd.seed(id)
        self._probHead = prob_head  # probability of flipping a head
        self._countWins = 0  # number of wins, set to 0 to begin

    def simulate(self, n_of_flips):

        count_tails = 0  # number of consecutive tails so far, set to 0 to begin

        # flip the coin 20 times
        for i in range(n_of_flips):

            # in the case of flipping a heads
            if self._rnd.random_sample() < self._probHead:
                if count_tails >= 2:  # if the series is ..., T, T, H
                    self._countWins += 1  # increase the number of wins by 1
                count_tails = 0  # the tails counter needs to be reset to 0 because a heads was flipped

            # in the case of flipping a tails
            else:
                count_tails += 1  # increase tails count by one

    def get_reward(self):
        # calculate the reward from playing a single game
        return 100*self._countWins - 250


class SetOfGames:
    def __init__(self, prob_head, n_games):
        self._gameRewards = [] # create an empty list where rewards will be stored

        # simulate the games
        for n in range(n_games):
            # create a new game
            game = Game(id=n, prob_head=prob_head)
            # simulate the game with 20 flips
            game.simulate(20)
            # store the reward
            self._gameRewards.append(game.get_reward())


    def get_ave_reward(self):
        """ returns the average reward from all games"""
        return sum(self._gameRewards) / len(self._gameRewards)

    def get_max(self):
        self._high = 0
        for val in self._gameRewards:
            if val>self._high:
                self._high = val
        return self._high

    def get_min(self):
        self._low = 0

        for val in self._gameRewards:
            if val<self._low:
                self._low = val
        return self._low

    def get_loss_prob(self):
        self._losscount=0
        for value in self._gameRewards:
            if value> 0:
                self._losscount +=1
        return self._losscount/len(self._gameRewards)

    def list_of_rewards(self):
        return self._gameRewards


# run trail of 1000 games to calculate expected reward
games = SetOfGames(prob_head=0.5, n_games=1000)

# plot the histogram
Fig.graph_histogram(
    observations=games.list_of_rewards(),
    title='Histogram of Rewards',
    x_label='Time',
    y_label='Reward Count')

# print the average reward
print('The expected probability that you lose money in this game after 1000 game simulations is ', games.get_loss_prob(),'.')
print("The max reward value was ", games.get_max(),'.')
print("The minimum reward value was ", games.get_min(),'.')


