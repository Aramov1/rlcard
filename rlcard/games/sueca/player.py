from .card import SuecaCard
from .utils import NUM_PLAYERS


class SuecaPlayer:

    def __init__(self, player_id: int, np_random):
        ''' Initialize a SuecaPlayer

        Args:
            player_id (int): identifier for the player, in range(4)
        '''
        if player_id < 0 or player_id >= NUM_PLAYERS:
            raise Exception(f'SuecaPlayer has invalid player_id: {player_id}')

        self.np_random = np_random
        self.player_id: int = player_id
        self.hand = []
        self.played_cards = []

    @property
    def team_id(self) -> int:
        ''' Partners sit opposite each other. 
        Seats 0 and 2 are one team and seats 1 and 3 are the other
        '''
        return self.player_id % 2

    def remove_card_from_hand(self, card: SuecaCard):
        self.hand.remove(card)

    def __str__(self):
        return f'P{self.player_id}'
