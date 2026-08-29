from .card import SuecaCard
from .player import SuecaPlayer


class SuecaDealer:
    ''' Shuffles the deck and distributes the cards for all 4 players
    '''

    def __init__(self, np_random):
        self.np_random = np_random

        # Get pre-ordered deck, 
        self.shuffled_deck = SuecaCard.get_deck() 

        # Shuffle deck
        self.np_random.shuffle(self.shuffled_deck) 

        # Copy of the shuffled deck to the deck to be delivered among the players. 
        # and keep a coppy of the shuffled deck
        self.stock_pile = self.shuffled_deck.copy() # Copy of the shuffled deck to be used for dealing

    def deal_cards(self, player: SuecaPlayer, num: int):
        ''' Deal some cards from the stock pile to one player

        Args:
            player (SuecaPlayer): The player to deal to
            num (int): The number of cards to deal
        '''
        for _ in range(num):
            player.hand.append(self.stock_pile.pop())
