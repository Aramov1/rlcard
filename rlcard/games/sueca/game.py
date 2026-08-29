from copy import deepcopy
from typing import List

import numpy as np

from .card import SuecaCard
from .judger import REWARD_SCHEMES, SuecaJudger
from .round import SuecaRound
from .utils import NUM_CARDS, NUM_PLAYERS, NUM_TRICKS


class SuecaGame:
    def __init__(self, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = NUM_PLAYERS
        self.judger = SuecaJudger(game=self)
        self.round = None
        self.history = []

        # game configurations, see rlcard/envs/sueca.py
        self.dealer_id = None         # None deals from a random seat every hand
        self.reward_scheme = 'points' # one of REWARD_SCHEMES, see SuecaJudger.judge_payoffs
        self.reward_shaping_coefficient = 0.0      # only read when reward_scheme = 'shaped'; 
                                      # 0, then the pays happen only at the end (identical to 'points'),
                                      # 1, then the pays happen trick by trick (the cumulative sum of the reward is still the same as 'points', but distributed in multiple steps)

    def configure(self, game_config):
        ''' Read game specific parameters '''

        self.dealer_id = game_config['game_dealer_id']

        self.reward_scheme = game_config['game_reward_scheme']
        if self.reward_scheme not in REWARD_SCHEMES:
            raise ValueError(
                f'Unknown reward scheme {self.reward_scheme!r}, expected one of {REWARD_SCHEMES}')

        self.reward_shaping_coefficient = game_config['game_reward_shaping_coefficient']
        if not 0.0 <= self.reward_shaping_coefficient <= 1.0:
            raise ValueError(
                f'game_reward_shaping_coefficient must be in [0, 1], got {self.reward_shaping_coefficient}: it is the '
                f'share of the hand paid trick by trick, the rest being paid at the end')
        if self.reward_shaping_coefficient and self.reward_scheme != 'shaped':
            raise ValueError(
                f"game_reward_shaping_coefficient ({self.reward_shaping_coefficient}) needs game_reward_scheme 'shaped', "
                f'got {self.reward_scheme!r}: the per-trick rewards sum to the points the team '
                f'took, so under any other scheme they would not add up to the payoff of the '
                f'hand')

    def init_game(self):
        ''' Start a new hand of Sueca
        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): The id of the player that leads the first trick
        '''
        # Determine card dealer
        dealer_id = self.dealer_id
        if dealer_id is None:
            dealer_id = int(self.np_random.randint(self.num_players))

        # Create a new Sueca round
        self.round = SuecaRound(dealer_id=dealer_id, np_random=self.np_random)
        self.history = []
        current_player_id = self.round.current_player_id

        # Return initial game state, 
        return self.get_state(current_player_id), current_player_id

    def step(self, action: SuecaCard):
        ''' Play one card

        Args:
            action (SuecaCard): The card played by the current player

        Returns:
            (tuple): Tuple containing:

                (dict): The next state
                (int): The id of the next player
        '''
        if self.allow_step_back:
            # Seeding the memo with the random state keeps it shared instead of
            # copied: the round draws no randomness once the hand is dealt, and
            # a search that steps back often should not pay for copying it.
            self.history.append(deepcopy(self.round, {id(self.np_random): self.np_random}))

        self.round.play_card(action)
        next_player_id = self.round.current_player_id
        return self.get_state(next_player_id), next_player_id

    def step_back(self):
        ''' Take back the card played most recently

        Returns:
            (bool): False if there is no move to take back
        '''
        if not self.history:
            return False
        self.round = self.history.pop()
        return True

    def get_state(self, player_id: int):
        ''' Everything player_id is allowed to know about the current state of the game

            Only this player's own hand plus public information is included:
            the cards played so far, the trump card (which the dealer showed
            when the cards where dealt) and the suits players have been seen to be void
            in. The lists are copied so that stored states are not mutated by
            later play.

        Args:
            player_id (int): The id of the player

        Returns:
            (dict): The state seen by that player
        '''
        game_round = self.round
        player = game_round.players[player_id]
        is_current_player = player_id == game_round.current_player_id
        legal_actions = self.get_legal_actions() if is_current_player and not self.is_over() else []
        state = {
            'player_id': player_id,
            'current_player': game_round.current_player_id,
            'num_players': self.num_players,
            'dealer_id': game_round.dealer_id,
            'hand': list(player.hand),
            'trick': list(game_round.current_trick),
            'lead_suit': game_round.lead_suit,
            'trump_suit': game_round.trump_suit,
            'trump_card': game_round.trump_card,
            'played_cards': [list(other.played_cards) for other in game_round.players],
            'known_voids': [list(voids) for voids in game_round.known_voids],
            'team_points': list(game_round.team_points),
            'trick_index': len(game_round.trick_history),
            'num_cards_left': [len(other.hand) for other in game_round.players],
            'legal_actions': legal_actions,
        }
        return state

    def get_payoffs(self):
        ''' The payoff of every seat at the end of the hand. 
            Defined through 'game_reward_scheme' in the game config, see SuecaJudger.judge_payoffs.

        Returns:
            (numpy.array): A payoff for each of the four seats
        '''
        return self.judger.judge_payoffs(self.round.team_points, self.reward_scheme)

    def get_step_rewards(self):
        ''' The reward of every transition of the finished hands

            Every player plays exactly one card per trick, so entry [i][k] is
            the reward of the k-th transition of player i.

            Under every scheme except 'shaped', the whole payoff is paid on the
            last transition and the rest are zero. Under 'shaped',
            ``game_reward_shaping_coefficient`` splits the hand between an immediate reward
            paid as each trick is taken and the delayed payoff of the whole
            hand::

                rewards[i][k]  = reward_shaping_coefficient * trick_points[k] if my team took it
                rewards[i][-1] += (1 - reward_shaping_coefficient) * payoff[i]

        Returns:
            (numpy.array): An array of shape (num_players, NUM_TRICKS)
        '''
        rewards = np.zeros((self.num_players, NUM_TRICKS))
        reward_shaping_coefficient = self.reward_shaping_coefficient
        if reward_shaping_coefficient:
            for trick_index, (winner_id, points) in enumerate(self.round.trick_history):
                for player_id in range(self.num_players):
                    if player_id % 2 == winner_id % 2:
                        rewards[player_id][trick_index] = reward_shaping_coefficient * points
        rewards[:, -1] += (1 - reward_shaping_coefficient) * self.get_payoffs()
        return rewards

    def get_legal_actions(self) -> List[SuecaCard]:
        ''' The cards the current player may legally play '''
        return self.judger.get_legal_actions()

    def is_over(self) -> bool:
        ''' True if the hand is over, i.e. all tricks have been played '''
        return self.round is not None and self.round.is_over()

    def get_player_id(self) -> int:
        return self.round.current_player_id

    def get_num_players(self) -> int:
        return self.num_players

    @staticmethod
    def get_num_actions() -> int:
        ''' One action per card of the deck '''
        return NUM_CARDS
