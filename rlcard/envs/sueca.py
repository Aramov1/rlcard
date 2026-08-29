''' Sueca environment
'''

from collections import OrderedDict

import numpy as np

from rlcard.envs import Env
from rlcard.games.sueca import Game
from rlcard.games.sueca.card import SuecaCard
from rlcard.games.sueca.utils import (
    NUM_CARDS,
    NUM_PLAYERS,
    NUM_TRICKS,
    SUIT_INDEX,
    SUITS,
    TOTAL_POINTS,
    trick_points,
    trick_winner,
)

DEFAULT_GAME_CONFIG = {
    'game_dealer_id': None,          # Seat that deals the hand. None picks a random seat every hand.
    'game_reward_scheme': 'points',  # How a finished hand is rewarded: 'points', 'victories', 'victories_sym' or 'shaped'. See SuecaJudger.judge_payoffs.
    'game_reward_shaping_coefficient': 0.0,     # The credit-shaping coefficient. Under 'shaped', the share of the hand
                                                # paid trick by trick instead of at the end, in [0, 1]; refused non-zero
                                                # under any other scheme. See SuecaGame.get_step_rewards.
}

# The observation. Everything is encoded relative to the player about to act,
# because the four seats of Sueca are symmetric: relative seat 1 is the next
# player, 2 is the partner and 3 is the previous player. 
STATE_BLOCKS = (
    ('hand', NUM_CARDS),                                # my cards
    ('current_trick', NUM_PLAYERS * NUM_CARDS),         # the trick in progress, by relative seat
    ('played_cards', NUM_PLAYERS * NUM_CARDS),          # every card each seat has played so far
    ('unseen_cards', NUM_CARDS),                        # cards still hidden in the other three hands
    ('trump_suit', len(SUITS)),
    ('trump_card', NUM_CARDS),                          # the card the dealer turned up
    ('lead_suit', len(SUITS)),                          # all zero when I am leading
    ('known_voids', (NUM_PLAYERS - 1) * len(SUITS)),    # suits the others have been seen void in
    ('team_points', 2),                                 # [mine, theirs], normalized
    ('trick_index', NUM_TRICKS),                        # all zero once the hand is over
    ('position_in_trick', NUM_PLAYERS),                 # leading, 2nd, 3rd or 4th to play
    ('dealer_seat', NUM_PLAYERS),                       # relative seat holding the trump card
    ('trick_winner', 3),                                # nobody / opponent / partner is taking it
    ('trick_points', 1),                                # points already on the table, normalized
)

STATE_SHAPE_SIZE = sum(size for _, size in STATE_BLOCKS)


class SuecaEnv(Env):
    ''' Sueca Environment
    '''

    def __init__(self, config):
        self.name = 'sueca'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.state_shape = [[STATE_SHAPE_SIZE] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def get_payoffs(self):
        ''' Get the payoffs of players.

        Returns:
            (numpy.array): A list of payoffs for each player.
        '''
        return self.game.get_payoffs()

    def get_step_rewards(self):
        ''' Get the reward of every transition of the finished hand.

        Returns:
            (numpy.array): An array of shape (num_players, NUM_TRICKS).
        '''
        return self.game.get_step_rewards()

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        return self.game.round.get_perfect_information()

    def _extract_state(self, state):
        ''' Extract useful information from state for RL.

        Args:
            state (dict): The raw state

        Returns:
            (dict): The extracted state
        '''
        legal_actions = OrderedDict({card.card_id: None for card in state['legal_actions']})
        extracted_state = {
            'obs': _encode_state(state),
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': list(state['legal_actions']),
            'action_record': self.action_recorder,
        }
        return extracted_state

    def _decode_action(self, action_id):
        ''' Decode Action id to the action in the game.

        Args:
            action_id (int): The id of the action

        Returns:
            (SuecaCard): The card that will be passed to the game engine.
        '''
        return SuecaCard.card(action_id)

    def _get_legal_actions(self):
        ''' Get all legal actions for current state.

        Returns:
            (OrderedDict): An OrderedDict of legal actions' id.
        '''
        return OrderedDict({card.card_id: None for card in self.game.get_legal_actions()})


def _encode_state(state):
    ''' Turn the raw state of one player into the observation vector

    Args:
        state (dict): The raw state returned by SuecaGame.get_state

    Returns:
        (numpy.array): A vector of STATE_SHAPE_SIZE entries
    '''
    me = state['player_id']
    rep = []

    # 
    hand_rep = np.zeros(NUM_CARDS, dtype=np.float32)
    for card in state['hand']:
        hand_rep[card.card_id] = 1
    rep.append(hand_rep)

    trick_rep = [np.zeros(NUM_CARDS, dtype=np.float32) for _ in range(NUM_PLAYERS)]
    for player_id, card in state['trick']:
        trick_rep[(player_id - me) % NUM_PLAYERS][card.card_id] = 1
    rep += trick_rep

    played_rep = [np.zeros(NUM_CARDS, dtype=np.float32) for _ in range(NUM_PLAYERS)]
    for player_id, cards in enumerate(state['played_cards']):
        plane = played_rep[(player_id - me) % NUM_PLAYERS]
        for card in cards:
            plane[card.card_id] = 1
    rep += played_rep

    unseen_rep = np.ones(NUM_CARDS, dtype=np.float32)
    for card in state['hand']:
        unseen_rep[card.card_id] = 0
    for cards in state['played_cards']:
        for card in cards:
            unseen_rep[card.card_id] = 0
    rep.append(unseen_rep)

    trump_suit_rep = np.zeros(len(SUITS), dtype=np.float32)
    trump_suit_rep[SUIT_INDEX[state['trump_suit']]] = 1
    rep.append(trump_suit_rep)

    trump_card_rep = np.zeros(NUM_CARDS, dtype=np.float32)
    trump_card_rep[state['trump_card'].card_id] = 1
    rep.append(trump_card_rep)

    lead_suit_rep = np.zeros(len(SUITS), dtype=np.float32)
    if state['lead_suit'] is not None:
        lead_suit_rep[SUIT_INDEX[state['lead_suit']]] = 1
    rep.append(lead_suit_rep)

    voids_rep = np.zeros((NUM_PLAYERS - 1, len(SUITS)), dtype=np.float32)
    for relative_seat in range(1, NUM_PLAYERS):
        voids = state['known_voids'][(me + relative_seat) % NUM_PLAYERS]
        for suit_index, is_void in enumerate(voids):
            if is_void:
                voids_rep[relative_seat - 1][suit_index] = 1
    rep.append(voids_rep.flatten())

    my_team = me % 2
    points_rep = np.array(
        [state['team_points'][my_team], state['team_points'][1 - my_team]],
        dtype=np.float32,
    ) / TOTAL_POINTS
    rep.append(points_rep)

    trick_index_rep = np.zeros(NUM_TRICKS, dtype=np.float32)
    if state['trick_index'] < NUM_TRICKS:
        trick_index_rep[state['trick_index']] = 1
    rep.append(trick_index_rep)

    position_rep = np.zeros(NUM_PLAYERS, dtype=np.float32)
    position_rep[len(state['trick'])] = 1
    rep.append(position_rep)

    dealer_rep = np.zeros(NUM_PLAYERS, dtype=np.float32)
    dealer_rep[(state['dealer_id'] - me) % NUM_PLAYERS] = 1
    rep.append(dealer_rep)

    # I have not played into this trick yet, so the card that is winning always
    # belongs to one of the other three seats. Relative seat 2 is my partner,
    # relative seats 1 and 3 are the opponents.
    trick_winner_rep = np.zeros(3, dtype=np.float32)
    if not state['trick']:
        trick_winner_rep[0] = 1  # I am leading, the trick is nobody's yet
    else:
        winner_id, _ = trick_winner(state['trick'], state['trump_suit'])
        trick_winner_rep[2 if (winner_id - me) % NUM_PLAYERS == 2 else 1] = 1
    rep.append(trick_winner_rep)

    # Normalized by TOTAL_POINTS like team_points above, so that a single linear
    # layer can add the two: the score my team holds if I take this trick.
    trick_points_rep = np.array([trick_points([card for _, card in state['trick']])], dtype=np.float32) / TOTAL_POINTS
    rep.append(trick_points_rep)

    return np.concatenate(rep)


def reorganize_sueca(trajectories, step_rewards):
    ''' Reorganize the trajectory to make it RL friendly

        Same contract as rlcard.utils.reorganize, except that the reward of
        each transition is read from step_rewards instead of being the payoff
        on the last transition and zero everywhere else. Every player makes
        exactly one decision per trick, so the k-th transition of a player is
        the k-th trick and step_rewards[player][k] is its reward.

    Args:
        trajectories (list): A list of trajectories, as returned by env.run
        step_rewards (numpy.array): A (num_players, num_tricks) array of
            rewards, as returned by SuecaEnv.get_step_rewards

    Returns:
        (list): New trajectories that can be fed into RL algorithms.
    '''
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]

    for player in range(num_players):
        num_transitions = (len(trajectories[player]) - 1) // 2
        for index in range(num_transitions):
            done = index == num_transitions - 1
            transition = trajectories[player][2 * index:2 * index + 3].copy()
            transition.insert(2, step_rewards[player][index])
            transition.append(done)
            new_trajectories[player].append(transition)
    return new_trajectories
