from typing import List, Optional, Tuple

from .card import SuecaCard
from .dealer import SuecaDealer
from .player import SuecaPlayer
from .utils import (
    NUM_PLAYERS,
    NUM_TRICKS,
    SUIT_INDEX,
    SUITS,
    trick_points,
    trick_winner,
    victory_points,
)


class SuecaRound:
    ''' A whole hand of Sueca: the deal followed by ten consecutive tricks

        Seats are numbered in playing order, such that the next player to play is
        always ``(current_player_id + 1) % 4``
    '''

    def __init__(self, dealer_id: int, np_random):
        ''' Deal the ten cards to each player and turn the trump card face up
        Args:
            dealer_id (int): The dealer's seat id,
            np_random: The environment local random state
        '''
        self.np_random = np_random
        self.dealer_id = dealer_id

        # Create dealer
        self.dealer = SuecaDealer(self.np_random)

        # Create players
        self.players = [SuecaPlayer(player_id=player_id, np_random=self.np_random)
                        for player_id in range(NUM_PLAYERS)
        ]

        # Deal ten cards to each player
        for seat in range(1, NUM_PLAYERS + 1):
            self.dealer.deal_cards(player=self.players[(dealer_id + seat) % NUM_PLAYERS], num=NUM_TRICKS)

        self.trump_card = self.players[dealer_id].hand[-1]
        self.trump_suit = self.trump_card.suit

        # First player to start is the player right after the dealer
        self.current_player_id = (dealer_id + 1) % NUM_PLAYERS

        self.current_trick = []
        self.trick_history = []  # (winner_id, points) per completed trick
        self.team_points = [0, 0]

        # known_voids[player_id][suit_index] is True once that player has been
        # seen failing to follow a lead suit. This is public information.
        self.known_voids: List[List[bool]] = [[False] * len(SUITS) for _ in range(NUM_PLAYERS)]

    @property
    def lead_suit(self) -> Optional[str]:
        ''' The suit played by the player starting the trick, or None if no card has been played yet '''
        return self.current_trick[0][1].suit if self.current_trick else None

    def get_current_player(self) -> SuecaPlayer:
        return self.players[self.current_player_id]

    def is_over(self) -> bool:
        return len(self.trick_history) == NUM_TRICKS

    def play_card(self, card: SuecaCard):
        ''' Play a card for the current player. If the player is the last to play, 
        finish the trick and update the points and current player accordingly
        
        Args:
            card (SuecaCard): The card to play. It must be legal; see SuecaJudger
        '''
        player = self.players[self.current_player_id]
        lead_suit = self.lead_suit

        if lead_suit is not None and card.suit != lead_suit:
            # everybody at the table saw this player fail to follow suit
            self.known_voids[player.player_id][SUIT_INDEX[lead_suit]] = True

        player.remove_card_from_hand(card)
        player.played_cards.append(card)
        self.current_trick.append((player.player_id, card))

        if len(self.current_trick) < NUM_PLAYERS:
            self.current_player_id = (self.current_player_id + 1) % NUM_PLAYERS
            return

        # The trick is complete: the highest trump wins, otherwise the highest
        # card of the lead suit.
        winner_id, _ = trick_winner(self.current_trick, self.trump_suit)

        points = trick_points([played_card for _, played_card in self.current_trick])
        self.team_points[winner_id % 2] += points
        self.trick_history.append((winner_id, points))
        self.current_trick = []
        self.current_player_id = winner_id  # the winner leads the next trick

    def get_perfect_information(self):
        ''' Everything about the current state, including the hidden hands

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['dealer_id'] = self.dealer_id
        state['current_player'] = self.current_player_id
        state['trump_suit'] = self.trump_suit
        state['trump_card'] = self.trump_card
        state['lead_suit'] = self.lead_suit
        state['hands'] = [player.hand for player in self.players]
        state['played_cards'] = [player.played_cards for player in self.players]
        state['trick'] = self.current_trick
        state['trick_index'] = len(self.trick_history)
        state['trick_history'] = self.trick_history
        state['team_points'] = self.team_points
        state['victory_points'] = [victory_points(points) for points in self.team_points]
        state['known_voids'] = self.known_voids
        return state

    def print_scene(self):
        ''' Print the current state of the table, for debugging '''
        print(f'===== trick: {len(self.trick_history)} player: {self.get_current_player()} '
              f'trump: {self.trump_card} =====')
        print(f'dealer={self.players[self.dealer_id]}')
        for player in self.players:
            print(f'{player}: {[str(card) for card in player.hand]}')
        trick = [f'{self.players[player_id]}:{card}' for player_id, card in self.current_trick]
        print(f'trick: {trick}')
        print(f'points: team 0 = {self.team_points[0]}, team 1 = {self.team_points[1]}')
