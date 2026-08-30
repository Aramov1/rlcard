''' Score trained agents under one fixed protocol, and rank a whole sweep.

        python examples/sueca_rlv2/evaluate.py experiments/v2_stage_a
        python examples/sueca_rlv2/evaluate.py experiments/v2_final --num_games 2500
        python examples/sueca_rlv2/evaluate.py --ensemble experiments/v2_final/ens12 \
            --models 'experiments/v2_stage_d/30_seeds/*/model_best.pth'

    The protocol matters more than any single number in it, so it is fixed and
    applied to every agent identically.

    Two seatings. `solo` puts the agent in seat 0 only, so it carries a
    rule-based partner against two rule opponents: this asks whether it plays
    cards well. `team` puts it in seats 0 and 2, a full agent team against a
    full rule team, which is how a finished Sueca player is judged. The gap
    between them is informative in itself - it is what replacing a competent
    rule partner with a copy of the agent is worth.

    Paired deals. The deal depends only on the seed and not on what the agents
    do, so re-seeding before each agent replays an identical sequence of hands.
    Comparisons within a batch are therefore exactly paired, which removes the
    dominant source of variance: a hand that cannot be lost is handed to every
    agent equally.

    One yardstick for every agent: the points its team took out of the 120 in
    the deck. This is a property of the *hand*, not of the reward scheme the
    agent was trained under, which is what makes agents trained under different
    schemes comparable at all. Parity is exactly 60.0.

    Two reference lines, in every table. The rule agent under test fills all
    four seats with rule agents, so its score is seating-independent and its
    expected value is exactly 60.0; what it actually measures is the protocol's
    residual noise. The random agent is the floor, without which a reader cannot
    tell whether three points is a lot.

    Training-time scores are never used here: each run evaluates itself against
    its own training opponents, so a `shared` run is scored with two copies of
    itself at the table and a `solo` run with one, and the two cannot be ranked
    against each other.
'''
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch

import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.games.sueca.utils import HALF_POINTS, TOTAL_POINTS
from rlcard.models.sueca_rule_models import SuecaRuleAgentV1
from rlcard.utils import set_seed

SEATINGS = {'solo': (0,), 'team': (0, 2)}


def load_agent(model, num_actions=40):
    ''' Build the agent under test

    Args:
        model (str): 'rule', 'random', or the path of a saved agent
        num_actions (int): the size of the action space, for the random agent

    Returns:
        the agent
    '''
    if model == 'rule':
        return SuecaRuleAgentV1()
    if model == 'random':
        return RandomAgent(num_actions=num_actions)
    # these pickles are written by this repo, so unpickling them is the same
    # trust as running the script
    agent = torch.load(model, map_location='cpu', weights_only=False)
    if hasattr(agent, 'set_device'):
        agent.set_device(torch.device('cpu'))
    return agent


def play(job):
    ''' Play one agent, in one seating, on one seed's deals

    Args:
        job (tuple): (model path or name, seating name, seed, number of hands)

    Returns:
        (tuple): the job key and the points the agent's team took, per hand
    '''
    model, seating, seed, num_games = job
    torch.set_num_threads(1)

    # set_seed covers the global stream the rule and random agents draw their
    # tie-breaks from; the env config seeds the deal itself. Both are needed for
    # two agents to meet identical hands.
    set_seed(seed)
    env = rlcard.make('sueca', config={'seed': seed})
    agents = [SuecaRuleAgentV1() for _ in range(env.num_players)]
    under_test = load_agent(model, env.num_actions)
    for seat in SEATINGS[seating]:
        agents[seat] = under_test
    env.set_agents(agents)

    points = np.empty(num_games)
    for hand in range(num_games):
        env.run(is_training=False)
        points[hand] = env.game.round.team_points[0]
    return (model, seating), points


def summarise(points):
    ''' The three numbers reported for one cell

        A 60-60 draw counts as a hand *not* won. The cost of that convention is
        that it is not symmetric - two identical agents each score just under
        50%, short by the draw rate - which is why an all-rule table reads about
        49% rather than 50%. It is stated rather than corrected, because the
        alternative hides how often Sueca hands end level.

    Args:
        points (numpy.array): the points taken, per hand

    Returns:
        (tuple): mean points, half-width of the 95% interval, and the win rate
    '''
    return (float(points.mean()),
            float(1.96 * points.std() / np.sqrt(len(points))),
            float((points > HALF_POINTS).mean()))


def find_models(sweep_dir):
    ''' Every trained run under a sweep directory, labelled by its condition

    Args:
        sweep_dir (str): a stage directory, or one holding runs directly

    Returns:
        (dict): label -> model path
    '''
    paths = sorted(glob.glob(os.path.join(sweep_dir, '*', '*', 'model.pth')))
    if not paths:
        paths = sorted(glob.glob(os.path.join(sweep_dir, '*', 'model.pth')))
    return {os.path.relpath(os.path.dirname(p), sweep_dir): p for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sweep_dir', nargs='?', help='directory of trained runs to score')
    parser.add_argument('--num_games', type=int, default=2000, help='hands per seed')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--ensemble', type=str, default=None,
                        help='instead of scoring, build an ensemble into this directory')
    parser.add_argument('--models', type=str, default=None,
                        help='glob of member models, with --ensemble')
    args = parser.parse_args()

    if args.ensemble:
        from agents import EnsembleAgent
        members = [load_agent(p) for p in sorted(glob.glob(args.models))]
        os.makedirs(args.ensemble, exist_ok=True)
        torch.save(EnsembleAgent(members), os.path.join(args.ensemble, 'model.pth'))
        print(f'=== {len(members)} members -> {args.ensemble}/model.pth')
        return

    models = {**find_models(args.sweep_dir), 'rule': 'rule', 'random': 'random'}
    jobs = [(path, seating, seed, args.num_games)
            for path in models.values()
            for seating in SEATINGS
            for seed in args.seeds]
    print(f'=== {len(models)} agents x {len(SEATINGS)} seatings x {len(args.seeds)} seeds '
          f'x {args.num_games:,} hands', flush=True)

    pooled = {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=get_context('spawn')) as pool:
        for key, points in pool.map(play, jobs):
            pooled.setdefault(key, []).append(points)

    rows = []
    for label, path in models.items():
        cells = {s: summarise(np.concatenate(pooled[(path, s)])) for s in SEATINGS}
        rows.append((cells['solo'][0], label, cells))
    rows.sort(reverse=True)

    hands = len(args.seeds) * args.num_games
    print(f'\n{"agent":34s} {"solo":>16s} {"team":>16s}   {"won":>6s}')
    print('-' * 78)
    for _, label, cells in rows:
        solo, team = cells['solo'], cells['team']
        print(f'{label:34s} {solo[0]:8.2f}+-{solo[1]:<5.2f} {team[0]:8.2f}+-{team[1]:<5.2f} '
              f'  {team[2]:5.1%}')
    print(f'\npoints are out of {TOTAL_POINTS}, parity is {HALF_POINTS}.0, '
          f'{hands:,} paired hands per cell.')
    print('a 60-60 draw counts as a hand not won, so an all-rule table reads ~49%.')


if __name__ == '__main__':
    main()
