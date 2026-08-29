''' Sueca judger
'''

from typing import List

import numpy as np

from .card import SuecaCard
from .utils import NUM_PLAYERS, victory_points

# The ways a finished hand can be turned into a reward. See judge_payoffs.
# 'shaped' is 'points' paid partly trick by trick; the split lives in
# SuecaGame.get_step_rewards, so the payoff of the whole hand is the same.
REWARD_SCHEMES = ('points', 'victories', 'victories_sym', 'shaped')


class SuecaJudger:
    ''' Decides which cards may be played and how a finished hand is scored
    '''

    def __init__(self, game):
        self.game = game

    def get_legal_actions(self) -> List[SuecaCard]:
        ''' The cards the current player may legally play

            A player must follow the lead suit whenever able to; otherwise any
            card may be played, with no obligation to trump. Because only legal
            cards are ever offered, a "renuncia" cannot occur.

        Returns:
            (list): A list of SuecaCard, ordered by card_id
        '''
        game_round = self.game.round
        hand = game_round.get_current_player().hand
        legal_cards = hand
        lead_suit = game_round.lead_suit
        if lead_suit is not None:
            cards_of_lead_suit = [card for card in hand if card.suit == lead_suit]
            if cards_of_lead_suit:
                legal_cards = cards_of_lead_suit
        return sorted(legal_cards, key=lambda card: card.card_id)

    @staticmethod
    def judge_payoffs(team_points, reward_scheme='points'):
        ''' The payoff of every seat at the end of a hand

            Four ways to score the same hand, all in the units the scheme is
            named for rather than normalized:

            'points'
                The points the seat's own team took, in [0, 120]. Half the
                deck is 60, so 61 is a bare win. This is the widest range of
                the four and the finest grained.
            'victories'
                The victory points the seat's own team won, one of 0, 1, 2 or
                4, and 0 for the losing team and for a 60-60 draw. This is how
                a real session is scored, and it ignores the margin inside a
                tier: 61-59 and 90-30 both pay 1.
            'victories_sym'
                The same tiers made zero sum by subtracting the opponents',
                so a lost hand is paid negatively rather than merely not
                rewarded.
            'shaped'
                Identical to 'points' here. The scheme differs only in *when*
                the hand is paid, which is decided in
                SuecaGame.get_step_rewards, so the total of a hand is the
                same.

            Only 'victories_sym' is zero sum. The other three are non-negative
            and pay both teams for the same hand, which is deliberate: the
            report defines them that way.

        Args:
            team_points (list): The points made by team 0 and team 1
            reward_scheme (str): One of REWARD_SCHEMES

        Returns:
            (numpy.array): A payoff for each of the four seats
        '''
        if reward_scheme not in REWARD_SCHEMES:
            raise ValueError(
                f'Unknown reward scheme {reward_scheme!r}, expected one of {REWARD_SCHEMES}')

        victories = SuecaJudger.judge_victories(team_points)
        payoffs = np.zeros(NUM_PLAYERS)
        for player_id in range(NUM_PLAYERS):
            team_id = player_id % 2
            if reward_scheme == 'victories':
                payoffs[player_id] = victories[team_id]
            elif reward_scheme == 'victories_sym':
                payoffs[player_id] = victories[team_id] - victories[1 - team_id]
            else:  # 'points' and the terminal half of 'shaped'
                payoffs[player_id] = team_points[team_id]
        return payoffs

    @staticmethod
    def judge_victories(team_points):
        ''' The victory points won by each team, as scored in a real session

        Args:
            team_points (list): The points made by team 0 and team 1

        Returns:
            (list): The victory points of team 0 and team 1
        '''
        return [victory_points(points) for points in team_points]
