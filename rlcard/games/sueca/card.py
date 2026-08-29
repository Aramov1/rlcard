from typing import List

from rlcard.games.base import Card
from .utils import CARD_POINTS, RANKS, SUITS


class SuecaCard(Card):
    ''' The Sueca is played with a 40 card deck. 
    Every card is uniquely identified by its 'card_id' in range(40). 
    '''

    suits = SUITS
    ranks = RANKS

    @staticmethod
    def card(card_id: int) -> 'SuecaCard':
        ''' Get the card with the given card_id '''
        return _deck[card_id]

    @staticmethod
    def get_deck() -> List['SuecaCard']:
        ''' Get a fresh copy of the 40 card deck, ordered by card_id '''
        return _deck.copy()

    @staticmethod
    def from_index(index: str) -> 'SuecaCard':
        ''' Get the card with the given suit-first index, e.g. 'SA' '''
        return _index_to_card[index]

    def __init__(self, suit: str, rank: str):
        ''' Initialize the class of SuecaCard

        Args:
            suit (str): The type of card
            rank (str): The rank of card
        '''
        super().__init__(suit=suit, rank=rank)
        self.card_id = len(RANKS) * SUITS.index(suit) + RANKS.index(rank)
        self.points = CARD_POINTS.get(rank, 0)

# Build the deck and the index lookup table for all 40 cards
_deck = [SuecaCard(suit=suit, rank=rank) for suit in SUITS for rank in RANKS]

# Build a lookup table from the suit-first index to the card object
_index_to_card = {card.get_index(): card for card in _deck}
