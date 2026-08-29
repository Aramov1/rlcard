''' Sueca rule models

    The rule-based player of Correia et al., "A Social Robot as a Card Game
    Player" (AIIDE-17), which the paper uses as the baseline every other player
    is measured against.
'''

import numpy as np

import rlcard
from rlcard.games.sueca.utils import RANKS, SUIT_INDEX, wins_trick
from rlcard.models.model import Model


class SuecaRuleAgentV1(object):
    ''' Sueca Rule agent version 1

        Start by identifying the highest cards of each allowed suit for the current play.
        It then assesses whether or not to play each card by checking: 
            (i) if it is the highest unplayed card of that suit would actually be winning the trick when played;
            (ii) if there are less than 5 cards from that same suit in its hand, except for the trump suit. 
        If both conditions are true, it plays one of its highest cards, otherwise it plays a random
        zero-value card.
    '''

    def __init__(self, np_random=None):
        ''' Initialize the rule agent

        Args:
            np_random: Optional random state used to break ties.
        '''
        self.use_raw = True
        self.np_random = np_random if np_random is not None else np.random

    @staticmethod
    def get_candidate_actions(state):
        ''' The cards the rule considers playing, before the breaking the tie

        Args:
            state (dict): The state dict handed to the agent by the environment

        Returns:
            (list): A non-empty list of SuecaCard, ordered by card_id
        '''
        legal_actions = state['raw_legal_actions']
        raw_obs = state['raw_obs']
        hand = raw_obs['hand']
        trick = raw_obs['trick']
        trump_suit = raw_obs['trump_suit']

        # Every card anybody has played, including the trick in progress: the
        # game appends to played_cards as soon as a card is laid down.
        played = {card.card_id for cards in raw_obs['played_cards'] for card in cards}

        # The highest allowed card of each suit. When following suit there is
        # only one suit to consider; when leading or void there are up to four.
        highest_of_suit = {}
        for card in legal_actions:
            if card.suit not in highest_of_suit or card.card_id > highest_of_suit[card.suit].card_id:
                highest_of_suit[card.suit] = card

        candidates = []
        for card in highest_of_suit.values():
            # (i) every stronger card of this suit is already out of play. Cards
            # of this suit left in our own hand cannot fail the test, since this
            # is the highest one we hold.
            suit_end = len(RANKS) * (SUIT_INDEX[card.suit] + 1)
            if any(card_id not in played for card_id in range(card.card_id + 1, suit_end)):
                continue
            # (ii) we are not so long in this suit that somebody is likely void
            # in it and able to cut. Trumps cannot be cut, so they skip this.
            if card.suit != trump_suit and sum(1 for c in hand if c.suit == card.suit) >= 5:
                continue
            # and it has to win the trick as it stands, see the class docstring
            if not wins_trick(card, trick, trump_suit):
                continue
            candidates.append(card)

        if not candidates:
            # Discard instead. The cheapest legal cards are the zero-value ones
            # whenever the hand holds any; following suit can force a choice
            # between scoring cards only, and then the cheapest one is shed.
            min_points = min(card.points for card in legal_actions)
            candidates = [card for card in legal_actions if card.points == min_points]

        return sorted(candidates, key=lambda card: card.card_id)

    def step(self, state):
        ''' Predict the action given the raw state

        Args:
            state (dict): The state dict handed to the agent by the environment

        Returns:
            action (SuecaCard): The card to play
        '''
        candidates = self.get_candidate_actions(state)
        return candidates[self.np_random.randint(len(candidates))]

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []


class SuecaRuleModelV1(Model):
    ''' Sueca Rule Model version 1
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('sueca')

        rule_agent = SuecaRuleAgentV1()
        self.rule_agents = [rule_agent for _ in range(env.num_players)]

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.rule_agents

    @property
    def use_raw(self):
        ''' Indicate whether use raw state and action

        Returns:
            use_raw (boolean): True if using raw state and action
        '''
        return True
