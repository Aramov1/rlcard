''' Global Defines and helper functions for Sueca Game
'''

# Available suits: Spades, Hearts, Diamonds, Clubs
SUITS = ['S', 'H', 'D', 'C'] 

# Ranks ordered in ascending trick strength: 2 < 3 < 4 < 5 < 6 < Q < J < K < 7 < A
RANKS = ['2', '3', '4', '5', '6', 'Q', 'J', 'K', '7', 'A']

# Sueca card points. Ranks not listed here are worth nothing.
CARD_POINTS = {'A': 11, '7': 10, 'K': 4, 'J': 3, 'Q': 2}

SUIT_INDEX = {suit: index for index, suit in enumerate(SUITS)}

NUM_PLAYERS = 4
NUM_TRICKS = 10
NUM_CARDS = len(SUITS) * len(RANKS)  # 40

# Points dealt in a single hand: 4 suits x (11 + 10 + 4 + 3 + 2)
TOTAL_POINTS = len(SUITS) * sum(CARD_POINTS.values())  # 120

# A team needs more than half of the points on the table to win the hand.
HALF_POINTS = TOTAL_POINTS // 2  # 60


def trick_winner(trick, trump_suit) -> tuple:
    ''' Find  the card that is winning a trick at a given moment. 
        The highest trump wins; If no trumps are played, then the highest card of 
        the same suit as the first card playedwins. Cards of any other suit cannot win. 

    Args:
        trick (list): A non-empty list of (player_id, SuecaCard) in playing order
        trump_suit (str): The trump suit of the hand

    Returns:
        (tuple): The (player_id, SuecaCard) pair that is winning
    '''
    winner_id, winning_card = trick[0]
    for player_id, card in trick[1:]:
        if card.suit == winning_card.suit:

            #  Because card ids are ordered in increasing card value over the suit, then 
            #  two cards of the same suit can be compared using their card ids.
            if card.card_id > winning_card.card_id:
                winner_id, winning_card = player_id, card

        elif card.suit == trump_suit:
            winner_id, winning_card = player_id, card

    return winner_id, winning_card


def wins_trick(card, trick, trump_suit) -> bool:
    ''' Find whether playing a given card would win the trick until then

    Args:
        card (SuecaCard): The card being considered
        trick (list): The (player_id, SuecaCard) pairs played so far, possibly empty
        trump_suit (str): The trump suit of the hand

    Returns:
        (bool): True when leading, or when the card beats the card currently winning
    '''
    # if leading trick,then always wins the trick as no other card was played yet
    if not trick:
        return True
    
    _, winning_card = trick_winner(trick, trump_suit)
    if card.suit == winning_card.suit:
        return card.card_id > winning_card.card_id
    return card.suit == trump_suit


def trick_points(cards):
    ''' Count the points of a group of cards

    Args:
        cards (list): A list of SuecaCard

    Returns:
        (int): The sum of the point values of the cards
    '''
    return sum(CARD_POINTS.get(card.rank, 0) for card in cards) 


def victory_points(team_points):
    ''' Convert the points of a team for a givenhand into victory points

        61-90 points wins one victory, 91-119 wins two and 120 wins four. 
        Anything else, wins zero

    Args:
        team_points (int): The points made by a team in a hand

    Returns:
        (int): The victory points earned by that team
    '''
    if team_points == TOTAL_POINTS:
        return 4
    if team_points > 90:
        return 2
    if team_points > HALF_POINTS:
        return 1
    return 0
