''' Play a hand of Sueca, choosing who sits in each of the four seats
'''

import os

import rlcard
from rlcard.agents.human_agents.sueca_human_agent import HumanAgent, _print_action
from rlcard.agents.random_agent import RandomAgent
from rlcard.games.sueca.utils import NUM_PLAYERS
from rlcard.models.sueca_rule_models import SuecaRuleAgentV1
from rlcard.utils import set_seed

# Who sits in each seat. Seats 0 and 2 are one team, seats 1 and 3 the other,
# so your partner is always two seats away. Edit this list to change the table.
# Each entry is one of:
#   'human'   - you, choosing cards at the terminal
#   'rule'    - the rule-based player of Correia et al. (AIIDE-17)
#   'random'  - a uniformly random legal card
#   a model id in the zoo, e.g. 'sueca-rule-v1'
#   a path to a saved agent, e.g. 'experiments/sueca_dqn_result/model.pth'
SEATS = ['human', 'rule', 'random', 'rule']

# How many hands to play. 0 keeps dealing until you stop the script, so set this
# when nobody is human and there is no prompt between hands to interrupt.
NUM_HANDS = 0

# An int here makes both the deal and the bots reproducible.
SEED = None


def _load_agent(spec, env, seat):
    ''' Turn one entry of SEATS into an agent

        The last two branches mirror load_model in examples/evaluate.py, so a
        model that works there works here. The code is repeated rather than
        imported because examples/ is not a package.

    Args:
        spec (str): One entry of SEATS
        env (Env): The Sueca environment the agent will play in
        seat (int): The seat the agent will occupy, needed to pick an agent out
            of a model from the zoo

    Returns:
        (object): An agent with step, eval_step and use_raw
    '''
    if spec == 'human':
        return HumanAgent(env.num_actions)
    if spec == 'rule':
        return SuecaRuleAgentV1()
    if spec == 'random':
        return RandomAgent(num_actions=env.num_actions)
    if os.path.isfile(spec):  # a torch agent saved by examples/run_rl.py
        import torch
        agent = torch.load(spec)
        agent.set_device(torch.device('cpu'))
        return agent
    from rlcard import models  # a model in the model zoo
    return models.load(spec).agents[seat]


def _print_table(perspective, has_human):
    ''' Announce who is playing where, once, before the first hand '''
    print('>> Sueca. Seats 0 and 2 are one team, seats 1 and 3 are the other.')
    for seat, spec in enumerate(SEATS):
        note = ''
        if has_human:
            if seat == perspective:
                note = '  <- you'
            elif seat == (perspective + 2) % NUM_PLAYERS:
                note = '  <- your partner'
        print('   seat {}: {:<10}{}'.format(seat, spec, note).rstrip())
    print('')


def _victories(count):
    ''' "1 victory" rather than "1 victories" '''
    return '1 victory' if count == 1 else '{} victories'.format(count)


def _print_plays_since_your_turn(trajectories, perspective):
    ''' Print the cards played after the perspective seat acted for the last time

        A player only sees the table when it is their turn, so whatever happened
        between their last card and the end of the hand has not been shown yet.
    '''
    final_state = trajectories[perspective][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record) + 1):
        if action_record[-i][0] == state['player_id']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'plays ', end='')
        _print_action(pair[1])
        print('')


def _print_tricks(trick_history):
    ''' Print who took each trick. Only used when nobody watched the hand '''
    print('=============== The Tricks ==============')
    for index, (winner_id, points) in enumerate(trick_history):
        print('trick {:>2}: player {} takes {:>2} points'.format(index + 1, winner_id, points))


def _print_result(perfect_information, payoffs, perspective, has_human):
    ''' Print the score of the hand that has just finished '''
    team_points = perfect_information['team_points']
    victory_points = perfect_information['victory_points']
    my_team = perspective % 2

    print('===============     Result     ===============')
    if has_human:
        print('Your team: {} points, {}'.format(
            team_points[my_team], _victories(victory_points[my_team])))
        print('Opponents: {} points, {}'.format(
            team_points[1 - my_team], _victories(victory_points[1 - my_team])))
        if payoffs[perspective] > 0:
            print('You win!')
        elif payoffs[perspective] < 0:
            print('You lose!')
        else:
            print('A draw!')
    else:
        for team in range(2):
            print('Team {} (seats {} and {}): {} points, {}'.format(
                team, team, team + 2, team_points[team], _victories(victory_points[team])))
        if team_points[0] == team_points[1]:
            print('A draw!')
        else:
            print('Team {} wins the hand!'.format(int(team_points[1] > team_points[0])))
    print('')


def main():
    if len(SEATS) != NUM_PLAYERS:
        raise ValueError('SEATS must name one agent per seat, so {} of them, not {}'.format(
            NUM_PLAYERS, len(SEATS)))

    set_seed(SEED)
    env = rlcard.make('sueca', config={'seed': SEED})
    agents = [_load_agent(spec, env, seat) for seat, spec in enumerate(SEATS)]
    env.set_agents(agents)

    human_seats = [seat for seat, agent in enumerate(agents) if isinstance(agent, HumanAgent)]
    has_human = bool(human_seats)
    # Everything is reported from one seat's point of view: yours if you are at
    # the table, otherwise seat 0 so an all-bot table still prints a full hand.
    perspective = human_seats[0] if has_human else 0

    _print_table(perspective, has_human)

    hand = 0
    while NUM_HANDS == 0 or hand < NUM_HANDS:
        hand += 1
        print('>> Start hand {}'.format(hand))

        trajectories, payoffs = env.run(is_training=False)
        perfect_information = env.get_perfect_information()

        if has_human:
            _print_plays_since_your_turn(trajectories, perspective)
        else:
            _print_tricks(perfect_information['trick_history'])

        _print_result(perfect_information, payoffs, perspective, has_human)

        if has_human:
            input("Press any key to continue...")


if __name__ == '__main__':
    main()
