import numpy as np
import scr.SamplePathClass as PathCls
import scr.FigureSupport as figureLibrary
import scr.StatisticalClasses as Stat
#hmwk



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
        self._reward=100*self._countWins - 250
        return self._reward


    def get_probability_loss(self):
        count_loss = 0
        if self._reward < 0:
            count_loss = 1
        elif self._reward >= 0:
            count_loss = 0
        return count_loss


class SetOfGames:

    def __init__(self, ids, prob_head, n_games):
        self._ids=ids
        self._gameRewards = [] # create an empty list where rewards will be stored
        self._probLoss =[] #create an empty list where losses will be stored

    # simulate the games
        for n in range(n_games):
            # create a new game
            game = Game(id=n, prob_head=prob_head)
            # simulate the game with 20 flips
            game.simulate(20)
            # store the reward
            self._gameRewards.append(game.get_reward())
            # store the probability of loss
            self._probLoss.append(game.get_probability_loss())


    def simulate (self):
        return OutcomesSetOfGames (self)


    def get_rewards (self):
        return self._gameRewards


    def get_prob_loss (self):
        return self._probLoss


class OutcomesSetOfGames:
    def __init__(self, simulated_set_of_games):
        self._simulatedGameSets = simulated_set_of_games


        # summary statistics on game rewards
        self._sumStat_gameRewards = Stat.SummaryStat('Rewards of Game', self._simulatedGameSets.get_rewards())


        # summary statistic on probability of loss
        self._sumStat_probLoss = Stat.SummaryStat('Loss (probability)', self._simulatedGameSets.get_prob_loss())


    def get_reward_list(self):
        """ returns all the rewards from all games to later be used for creation of histogram """
        return self._simulatedGameSets.get_rewards()


    def get_ave_reward(self):
        """ returns the average reward from all games"""
        return self._sumStat_gameRewards.get_mean()


    def get_probability_loss(self):
        """ returns the average reward from all games"""
        return self._sumStat_probLoss.get_mean()


    def get_CI_Game_Rewards(self, alpha):
        return self._sumStat_gameRewards.get_t_CI(alpha)


    def get_CI_Prob_Loss(self, alpha):
        """
        :param alpha: confidence level
        :return: t-based confidence interval
        """
        return self._sumStat_probLoss.get_t_CI(alpha)


class MultiSetofGames:
    """ simulates multiple sets of games with different parameters """


    def __init__(self, ids, prob_head, n_games):


        self._ids = ids
        self._probHead=prob_head
        self._setofgamessize=n_games


        self._gameRewards = []      # two dimensional list of reward from each simulated set of games
        self._aveGameRewards = []   # list of mean rewrad for each simulated set of games
        self._probLoss = []      # two dimensional list of the probability of loss from each simulated set of games
        self._sumStat_meanGameRewards = None
        self._sumStat_probLoss = None


    def simulate(self):
        """ simulates all sets of games """


        for i in range(len(self._ids)):
            # create a multiset
            setofgames = SetOfGames(ids=self._ids[i], prob_head=self._probHead[i], n_games=self._setofgamessize[i])
            # simulate the multiset
            output = setofgames.simulate()
            # store games outcomes from this set of games
            self._gameRewards.append(setofgames.get_rewards())
            # store average survival time for this set of games
            self._aveGameRewards.append(output.get_ave_reward())
            # store the probability of losing for this set of games
            self._probLoss.append(setofgames.get_prob_loss())


        # after simulating all set of games
        # summary statistics of mean average reward
        self._sumStat_meanGameRewards = Stat.SummaryStat('Mean Game Reward', self._aveGameRewards)


        #summary statistics of the average probability of loss
        self._sumStat_probLoss=Stat.SummaryStat('Loss (probability)', self._probLoss)


    def get_mean_reward(self, set_of_games_index):


        return self._aveGameRewards[set_of_games_index]


    def get_overall_mean_reward(self):
        return self._sumStat_meanGameRewards.get_mean()


# problem 1
trial = SetOfGames(ids=1, prob_head=0.5, n_games=1000)
SimulateOutcomes = trial.simulate()


print ("Problem 1")
print (" ")
print("The average expected reward is:", SimulateOutcomes.get_ave_reward())
print("After 1000 simulations, the 95% confidence interval for the expected reward is: ",SimulateOutcomes.get_CI_Game_Rewards(0.05))
print (" ")
print("The average expected probability loss is:", SimulateOutcomes.get_probability_loss())
print("After 1000 simulations, the 95% confidence interval for the expected reward is: ",SimulateOutcomes.get_CI_Prob_Loss(0.05))
print (" ")
print (" ")
print("Problem 2")
print ("If you repeat this simulation 1000 times, on average 95% of the time this confidence interval of -31.8 to -20, will cover the true mean of rewards (-25.9).")
print ("If you repeat this simulation 1000 times, on average 95% of the time this confidence interval of of 0.58 to 0.63, will cover the true mean of rewards.")






#Problem 3 Owner
import numpy as np
import scr.SamplePathClass as PathCls
import scr.FigureSupport as figureLibrary
import scr.StatisticalClasses as Stat




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
        self._reward=250-100*self._countWins
        return self._reward


    def get_probability_loss(self):
        """ returns the probability of a loss """
        count_loss = 0
        if self._reward < 0:
            count_loss = 1
        elif self._reward >= 0:
            count_loss = 0
        return count_loss


class SetOfGames:
    def __init__(self, ids, prob_head, n_games):
        self._ids=ids
        self._gameRewards = [] # create an empty list where rewards will be stored
        self._probLoss =[] #create an empty list where losses will be stored


        # simulate the games
        for n in range(n_games):
            # create a new game
            game = Game(id=n, prob_head=prob_head)
            # simulate the game with 20 flips
            game.simulate(20)
            # store the reward
            self._gameRewards.append(game.get_reward())
            # store the probability of loss
            self._probLoss.append(game.get_probability_loss())


    def simulate (self):
        return OutcomesSetOfGames (self)


    def get_rewards (self):
        return self._gameRewards


    def get_prob_loss (self):
        return self._probLoss


class OutcomesSetOfGames:
    def __init__(self, simulated_set_of_games):
        self._simulatedGameSets = simulated_set_of_games


        # summary statistics on game rewards
        self._sumStat_gameRewards = Stat.SummaryStat('Rewards of Game', self._simulatedGameSets.get_rewards())


        # summary statistic on probability of loss
        self._sumStat_probLoss = Stat.SummaryStat('Loss (probability)', self._simulatedGameSets.get_prob_loss())


    def get_reward_list(self):
        """ returns all the rewards from all game to later be used for creation of histogram """
        return self._simulatedGameSets.get_rewards()


    def get_ave_reward(self):
        """ returns the average reward from all games"""
        return self._sumStat_gameRewards.get_mean()


    def get_probability_loss(self):
        """ returns the average reward from all games"""
        return self._sumStat_probLoss.get_mean()


    def get_CI_Game_Rewards(self, alpha):
        return self._sumStat_gameRewards.get_t_CI(alpha)


    def get_CI_Prob_Loss(self, alpha):
        """
        :param alpha: confidence level
        :return: t-based confidence interval
        """
        return self._sumStat_probLoss.get_t_CI(alpha)


class MultiSetofGames:
    """ simulates multiple sets of games with different parameters """


    def __init__(self, ids, prob_head, n_games):


        self._ids = ids
        self._probHead=prob_head
        self._setofgamessize=n_games


        self._gameRewards = []      # two dimensional list of reward from each simulated set of games
        self._aveGameRewards = []   # list of mean rewrad for each simulated set of games
        self._probLoss = []      # two dimensional list of the probability of loss from each simulated set of games
        self._sumStat_meanGameRewards = None
        self._sumStat_probLoss = None


    def simulate(self):
        """ simulates all sets of games """


        for i in range(len(self._ids)):
            # create a multiset
            setofgames = SetOfGames(ids=self._ids[i], prob_head=self._probHead[i], n_games=self._setofgamessize[i])
            # simulate the multiset
            output = setofgames.simulate()
            # store games outcomes from this set of games
            self._gameRewards.append(setofgames.get_rewards())
            # store average survival time for this set of games
            self._aveGameRewards.append(output.get_ave_reward())
            # store the probability of losing for this set of games
            self._probLoss.append(setofgames.get_prob_loss())


        # after simulating all set of games
        # summary statistics of mean average reward
        self._sumStat_meanGameRewards = Stat.SummaryStat('Mean Game Reward', self._aveGameRewards)


        #summary statistics of the average probability of loss
        self._sumStat_probLoss=Stat.SummaryStat('Probability of Loss', self._probLoss)


    def get_mean_reward(self, set_of_games_index):


        return self._aveGameRewards[set_of_games_index]


    def get_overall_mean_reward(self):
        return self._sumStat_meanGameRewards.get_mean()


# problem 1
trial = SetOfGames(ids=1, prob_head=0.5, n_games=1000)
SimulateOutcomes = trial.simulate()


print (" ")
print("Problem 3")
print("Because the sample size is so large for the owner, who is playing a virtually infinite number of games, he is a steady state so it would need a confidence interval")
print (" ")
print("The average expected reward is:", SimulateOutcomes.get_ave_reward())
print("After 1000 simulations, the 95% confidence interval for the expected reward is: ",SimulateOutcomes.get_CI_Game_Rewards(0.05))
print (" ")
print("The average expected probability loss is:", SimulateOutcomes.get_probability_loss())
print("After 1000 simulations, the 95% confidence interval for the expected reward is: ",SimulateOutcomes.get_CI_Prob_Loss(0.05))


#Problem 3 Gambler
import numpy as np
import scr.SamplePathClass as PathCls
import scr.FigureSupport as figureLibrary
import scr.StatisticalClasses as Stat




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
        self._reward=100*self._countWins - 250
        return self._reward


    def get_probability_loss(self):
        """ returns the probability of a loss """
        count_loss = 0
        if self._reward < 0:
            count_loss = 1
        elif self._reward >= 0:
            count_loss = 0
        return count_loss


class SetOfGames:
    def __init__(self, ids, prob_head, n_games):
        self._ids=ids
        self._gameRewards = [] # create an empty list where rewards will be stored
        self._probLoss =[] #create an empty list where losses will be stored


        # simulate the games
        for n in range(n_games):
            # create a new game
            game = Game(id=n, prob_head=prob_head)
            # simulate the game with 20 flips
            game.simulate(20)
            # store the reward
            self._gameRewards.append(game.get_reward())
            # store the probability of loss
            self._probLoss.append(game.get_probability_loss())


    def simulate (self):
        return OutcomesSetOfGames (self)


    def get_rewards (self):
        return self._gameRewards


    def get_prob_loss (self):
        return self._probLoss


class OutcomesSetOfGames:
    def __init__(self, simulated_set_of_games):
        self._simulatedGameSets = simulated_set_of_games


        # summary statistics on game rewards
        self._sumStat_gameRewards = Stat.SummaryStat('Game Rewards', self._simulatedGameSets.get_rewards())


        # summary statistic on probability of loss
        self._sumStat_probLoss = Stat.SummaryStat('Probability of Loss', self._simulatedGameSets.get_prob_loss())


    def get_reward_list(self):
        """ returns all the rewards from all games to later be used for creation of histogram """
        return self._simulatedGameSets.get_rewards()


    def get_ave_reward(self):
        """ returns the average reward from all games"""
        return self._sumStat_gameRewards.get_mean()


    def get_probability_loss(self):
        """ returns the average reward from all games"""
        return self._sumStat_probLoss.get_mean()


    def get_CI_Game_Rewards(self, alpha):
        return self._sumStat_gameRewards.get_t_CI(alpha)


    def get_CI_Prob_Loss(self, alpha):
        """
        :param alpha: confidence level
        :return: t-based confidence interval
        """
        return self._sumStat_probLoss.get_t_CI(alpha)


class MultiSetofGames:
    """ simulates multiple sets of games with different parameters """


    def __init__(self, ids, prob_head, n_games):


        self._ids = ids
        self._probHead=prob_head
        self._setofgamessize=n_games


        self._gameRewards = []      # two dimensional list of reward from each simulated set of games
        self._aveGameRewards = []   # list of mean rewrad for each simulated set of games
        self._probLoss = []      # two dimensional list of the probability of loss from each simulated set of games
        self._sumStat_meanGameRewards = None
        self._sumStat_probLoss = None


    def simulate(self):
        """ simulates all sets of games """


        for i in range(len(self._ids)):
            # create a multiset
            setofgames = SetOfGames(ids=self._ids[i], prob_head=self._probHead[i], n_games=self._setofgamessize[i])
            # simulate the multiset
            output = setofgames.simulate()
            # store games outcomes from this set of games
            self._gameRewards.append(setofgames.get_rewards())
            # store average survival time for this set of games
            self._aveGameRewards.append(output.get_ave_reward())
            # store the probability of losing for this set of games
            self._probLoss.append(setofgames.get_prob_loss())


        # after simulating all set of games
        # summary statistics of mean average reward
        self._sumStat_meanGameRewards = Stat.SummaryStat('Mean Game Reward', self._aveGameRewards)


        #summary statistics of the average probability of loss
        self._sumStat_probLoss=Stat.SummaryStat('Probability of Loss', self._probLoss)


    def get_mean_reward(self, set_of_games_index):


        return self._aveGameRewards[set_of_games_index]


    def get_overall_mean_reward(self):
        return self._sumStat_meanGameRewards.get_mean()


    def get_PI_mean_reward(self, alpha):
        return self._sumStat_meanGameRewards.get_PI(alpha)


def get_PI_reward(self, set_of_games_index, alpha):
    st = Stat.SummaryStat('', self._gameRewards[set_of_games_index])
    return st.get_PI(alpha)


# problem 3 Gambler


NUM_SET_GAMES = 1000
NUM_GAMES=10


trial = SetOfGames(ids=1, prob_head=0.5, n_games=10)
SimulateOutcomes = trial.simulate()


# calculating prediction interval for mean survival time
# create multiple cohorts
multiCohort = MultiSetofGames(
    ids=range(NUM_SET_GAMES),
    prob_head=[0.5]*NUM_SET_GAMES,
    n_games=[NUM_GAMES] * NUM_SET_GAMES,  # [REAL_POP_SIZE, REAL_POP_SIZE, ..., REAL_POP_SIZE]
)
# simulate all cohorts
multiCohort.simulate()
# print projection interval
print('95% PI of average reward ($)', multiCohort.get_PI_mean_reward(0.05))
