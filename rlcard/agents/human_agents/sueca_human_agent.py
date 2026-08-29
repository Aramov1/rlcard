from rlcard.games.sueca.utils import NUM_PLAYERS, NUM_TRICKS
from rlcard.utils.utils import elegent_form, print_card


class HumanAgent(object):
    ''' A human agent for Sueca. It can be used to play against trained models
    '''

    def __init__(self, num_actions):
        ''' Initilize the human agent

        Args:
            num_actions (int): the size of the ouput action space
        '''
        self.use_raw = True
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (SuecaCard): The card chosen by the human
        '''
        _print_state(state['raw_obs'], state['action_record'])
        action = int(input('>> You choose action (integer): '))
        while action < 0 or action >= len(state['legal_actions']):
            print('Action illegal...')
            action = int(input('>> Re-choose action (integer): '))
        return state['raw_legal_actions'][action]

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (SuecaCard): The card chosen by the human
        '''
        return self.step(state), {}


def _card_str(card):
    ''' The short form of a card, e.g. the ace of spades as "♠A" '''
    return elegent_form(card.get_index())


def _print_state(state, action_record):
    ''' Print out the state of a given player

    Args:
        state (dict): The raw state of the player about to act
        action_record (list): The (player_id, action) pairs played so far
    '''
    _action_list = []
    for i in range(1, len(action_record) + 1):
        if action_record[-i][0] == state['player_id']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'plays ', end='')
        _print_action(pair[1])
        print('')

    me = state['player_id']
    partner = (me + 2) % NUM_PLAYERS
    my_team = me % 2

    print('\n================ Trump ==================')
    print('{}, dealt to player {}'.format(_card_str(state['trump_card']), state['dealer_id']))

    print('=============== The Table ===============')
    print('You are player {}, your partner is player {}.'.format(me, partner))
    print('Trick {} of {}. Your team has {} points, the opponents have {}.'.format(
        state['trick_index'] + 1, NUM_TRICKS,
        state['team_points'][my_team], state['team_points'][1 - my_team]))

    print('============== Current Trick ============')
    if state['trick']:
        print('  '.join('player {}: {}'.format(player_id, _card_str(card))
                        for player_id, card in state['trick']))
        print_card([card for _, card in state['trick']])
    else:
        print('Empty, you lead.')

    print('=============== Your Hand ===============')
    print('  '.join(_card_str(card) for card in sorted(state['hand'], key=lambda card: card.card_id)))

    print('========= Actions You Can Choose ========')
    print(',  '.join('{}: {}'.format(index, _card_str(card))
                     for index, card in enumerate(state['legal_actions'])))
    print('')


def _print_action(action):
    ''' Print out an action in a nice form

    Args:
        action (SuecaCard): The card that was played
    '''
    print(_card_str(action), end='')
