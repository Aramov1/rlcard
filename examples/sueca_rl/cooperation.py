''' The two measurements that decide whether cooperation emerged.

    Points alone cannot answer it: two individually competent players score well
    without coordinating at all. So this module adds the two measurements that
    can, both run over agents that already exist and neither costing any
    training.

      assist_rate   Sueca's cooperative act is throwing a high card onto a trick
                    the *partner* is already taking, since the points go to the
                    team. Measured against the chance rate on the same
                    decisions, because a naive version of this metric scores a
                    random player above 50%.
      cross_play    If a pair learned a partner-specific convention, breaking
                    the pair should cost points. Re-pairs seat 0 of one run with
                    seat 2 of another, keeping individual skill and removing
                    only the shared history. This is what separates cooperation
                    from competence.

        python examples/sueca_rlv2/cooperation.py experiments/v2_coop
'''
import argparse
import os

import numpy as np
import torch

import rlcard
from rlcard.games.sueca.utils import NUM_PLAYERS, trick_winner, wins_trick
from rlcard.models.sueca_rule_models import SuecaRuleAgentV1
from rlcard.utils import set_seed

# Sueca partners sit opposite, so seats 0 and 2 are one team.
TEAM_SEATS = (0, 2)


def assist_rate(agent, num_games=2000, seed=7, seats=TEAM_SEATS):
    ''' How often the agent throws its points onto a trick the partner is taking

        In Sueca the points of a trick go to the team that takes it, so the
        cooperative play is to drop a high card - an Ace is 11, a Seven is 10 -
        onto a trick the *partner* has already won. A selfish or careless player
        keeps the card, and the ten points go nowhere.

        The rate is measured only on the decisions where the choice is genuine:
        the partner is currently winning the trick, no legal card of the agent's
        own would take it, and the legal cards do not all carry the same points.
        That last filter matters more than it looks. A forced play - one legal
        card - trivially plays the highest-value one and would score as a
        perfect assist, and a hand where every legal card is worth nothing
        offers no choice to observe. Counting those inflates every agent's rate
        towards each other and hides the effect.

        The rate is reported against the chance rate on the *same* decisions:
        for each one, the probability that a uniform choice among the legal
        cards would have assisted. An agent that assists at chance is not
        cooperating, it is playing arbitrarily, and without this baseline the
        two are indistinguishable - a random player scores over 50% on a naive
        version of this metric purely because high cards are common.

        Calibration, over 500 hands: the random player assists at 36.7% against
        a chance rate of 36.6%, a lift of +0.0%, which is what says the baseline
        is computed correctly. The rule player assists at 0.0% - it is not
        merely uncooperative, it is anti-cooperative on this axis, because when
        it cannot win a trick it deliberately plays its lowest-value card. Note
        what that implies: the rule player takes 59.8 points a hand while never
        assisting once. A high assist rate is therefore *not* a proxy for
        playing well, and this metric must be read beside the points, never
        instead of them.

    Args:
        agent: the agent under test
        num_games (int): hands to play
        seed (int): seeds the deal and the global stream, so every agent
            measured with the same seed meets identical hands
        seats (tuple): the seats the agent occupies

    Returns:
        (tuple): the assist rate, the chance rate on the same decisions, the
            number of qualifying decisions, and the mean points the team took
    '''
    set_seed(seed)
    env = rlcard.make('sueca', config={'seed': seed})
    players = [SuecaRuleAgentV1() for _ in range(NUM_PLAYERS)]
    for seat in seats:
        players[seat] = agent
    env.set_agents(players)

    assists = chances = 0
    chance_sum = 0.0
    points = []
    for _ in range(num_games):
        state, player_id = env.reset()
        while not env.is_over():
            acting = players[player_id]
            action, _ = acting.eval_step(state)
            if player_id in seats:
                assisted, counted, baseline = _score_decision(state, action, env)
                assists += assisted
                chances += counted
                chance_sum += baseline
            state, player_id = env.step(action, acting.use_raw)
        points.append(env.game.round.team_points[0])
    rate = assists / chances if chances else float('nan')
    chance = chance_sum / chances if chances else float('nan')
    return rate, chance, chances, float(np.mean(points))


def _score_decision(state, action, env):
    ''' Was this one decision an assist, did it qualify, and what was chance?

    Args:
        state (dict): the extracted state the agent acted on
        action: the action it chose, as a card id or a SuecaCard
        env (Env): the environment, for decoding the action

    Returns:
        (tuple): (1 if it assisted else 0, 1 if the decision qualified else 0,
            the probability a uniform legal choice would have assisted)
    '''
    raw = state['raw_obs']
    trick = raw['trick']
    if not trick:
        return 0, 0, 0.0  # leading: there is nothing to assist

    me = raw['player_id']
    trump = raw['trump_suit']
    winner_id, _ = trick_winner(trick, trump)
    if (winner_id - me) % NUM_PLAYERS != 2:
        return 0, 0, 0.0  # the partner is not the one taking it

    legal = raw['legal_actions']
    if len(legal) < 2:
        return 0, 0, 0.0  # forced: a single legal card is not a choice

    if any(wins_trick(card, trick, trump) for card in legal):
        return 0, 0, 0.0  # the agent could take the trick itself, a different decision

    values = [card.points for card in legal]
    best = max(values)
    if best == min(values):
        return 0, 0, 0.0  # every legal card is worth the same, so nothing to decide

    played = action if hasattr(action, 'points') else env._decode_action(action)
    baseline = values.count(best) / len(values)
    return int(played.points == best), 1, baseline


def cross_play(run_dirs, num_games=2000, seed=11):
    ''' Pair every seat-0 agent with every seat-2 agent and score the grid

        This is the test that separates cooperation from competence. A pair
        trained together shares a history; if that history produced a
        convention - a habit of leading a suit the partner is known to hold, of
        ducking in a particular spot - then breaking the pair should cost
        something. Re-pairing seat 0 of one run with seat 2 of another keeps the
        individual skill and removes only the shared history.

        Read the diagonal against the off-diagonal. Diagonal higher means the
        pairs coordinate; equal means each agent simply plays well and the
        partner is interchangeable, which is worth reporting as a negative
        rather than dressing up as cooperation.

    Args:
        run_dirs (list): duo run directories, each holding model_seat0.pth
            and model_seat2.pth
        num_games (int): hands per cell
        seed (int): the deals every cell is played on, so the grid is paired

    Returns:
        (tuple): the score grid as a nested list, and the run labels
    '''
    labels = [os.path.basename(d) for d in run_dirs]
    device = torch.device('cpu')
    seat0 = [torch.load(os.path.join(d, 'model_seat0.pth'),
                        map_location=device, weights_only=False) for d in run_dirs]
    seat2 = [torch.load(os.path.join(d, 'model_seat2.pth'),
                        map_location=device, weights_only=False) for d in run_dirs]

    grid = []
    for i, agent_a in enumerate(seat0):
        row = []
        for j, agent_b in enumerate(seat2):
            set_seed(seed)
            env = rlcard.make('sueca', config={'seed': seed})
            env.set_agents([agent_a, SuecaRuleAgentV1(), agent_b, SuecaRuleAgentV1()])
            points = []
            for _ in range(num_games):
                env.run(is_training=False)
                points.append(env.game.round.team_points[0])
            row.append(float(np.mean(points)))
            print(f'    {labels[i]} + {labels[j]}: {row[-1]:.2f}', flush=True)
        grid.append(row)
    return grid, labels


def report_cross_play(grid, labels):
    ''' Print the grid and the one number the experiment turns on

    Args:
        grid (list): the score grid from cross_play
        labels (list): the run labels
    '''
    size = len(labels)
    print('\n    seat0 \\ seat2  ' + ' '.join(f'{l[-7:]:>9s}' for l in labels))
    for i, label in enumerate(labels):
        print(f'    {label[-13:]:>13s}  ' + ' '.join(f'{grid[i][j]:9.2f}' for j in range(size)))

    diagonal = [grid[i][i] for i in range(size)]
    off = [grid[i][j] for i in range(size) for j in range(size) if i != j]
    if not off:
        return
    gap = float(np.mean(diagonal) - np.mean(off))
    print(f'\n    trained together (diagonal): {np.mean(diagonal):.2f}')
    print(f'    re-paired    (off-diagonal): {np.mean(off):.2f}')
    print(f'    cost of breaking the pair  : {gap:+.2f} points')
    verdict = ('pairs coordinate: breaking them costs real points'
               if gap > 0.5 else
               'no partner-specific coordination: the partner is interchangeable')
    print(f'    -> {verdict}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sweep_dir', type=str,
                        help='a cooperation sweep, holding 40_duo/, 40_shared/, 40_solo/')
    parser.add_argument('--num_games', type=int, default=2000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()

    from rlcard.agents.random_agent import RandomAgent
    print('=== assist rate: high card thrown onto a trick the partner is taking')
    print(f"    {'agent':10s} {'assist':>7s} {'chance':>7s} {'lift':>6s} {'n':>7s} {'points':>7s}")
    rows = [('rule', SuecaRuleAgentV1()), ('random', RandomAgent(num_actions=40))]
    for scheme in ('duo', 'shared', 'solo'):
        path = os.path.join(args.sweep_dir, f'40_{scheme}',
                            f'{scheme}_seed{args.seeds[0]}', 'model.pth')
        if os.path.exists(path):
            rows.append((scheme, torch.load(path, map_location='cpu', weights_only=False)))
    for label, agent in rows:
        rate, chance, n, points = assist_rate(agent, args.num_games)
        print(f'    {label:10s} {rate:7.1%} {chance:7.1%} {rate - chance:+6.1%} '
              f'{n:7d} {points:7.2f}', flush=True)

    duo_dir = os.path.join(args.sweep_dir, '40_duo')
    dirs = sorted(os.path.join(duo_dir, d) for d in os.listdir(duo_dir)
                  if os.path.exists(os.path.join(duo_dir, d, 'model_seat0.pth'))) \
        if os.path.isdir(duo_dir) else []
    if len(dirs) >= 2:
        print('\n=== cross-play')
        report_cross_play(*cross_play(dirs, args.num_games))
    else:
        print('\n=== cross-play needs at least two trained duo runs')


if __name__ == '__main__':
    main()
