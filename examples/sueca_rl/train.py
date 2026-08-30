''' Train Sueca agents. One engine, and every stage of the campaign as a config.

        python examples/sueca_rlv2/train.py a --workers 14     # the screen
        python examples/sueca_rlv2/train.py b --workers 8      # do the winners combine?
        python examples/sueca_rlv2/train.py d --workers 9      # many seeds
        python examples/sueca_rlv2/train.py coop --workers 9   # two independent learners
        CUDA_VISIBLE_DEVICES=0 python examples/sueca_rlv2/train.py c

    Run the sweeps on the CPU: `export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1`.
    The networks are tiny, so a run is bound by Python game logic rather than by
    matrix multiplication. One run is faster on the GPU (47 episodes/s against
    31 on a core), but processes serialise on the device, so twelve concurrent
    runs give 89 episodes/s on the GPU against 115 on the CPU. Stage C is a
    single long run and is the only stage that wants the GPU.

    Every stage is resumable: a run holding a model.pth is skipped, so an
    interrupted launch is continued by repeating the same command.
'''
import argparse
import contextlib
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from types import SimpleNamespace

import torch

import rlcard
from rlcard.envs.sueca import reorganize_sueca
from rlcard.models.sueca_rule_models import SuecaRuleAgentV1
from rlcard.utils import Logger, get_device, set_seed, tournament

from agents import PairAgent, without_buffers

ROOT = 'experiments'

DEFAULTS = dict(
    env='sueca',
    # --- the reward, which is what this study is about -------------------
    reward_scheme='points',              # points | victories | victories_sym | shaped
    reward_shaping_coefficient=0.0,      # share paid trick by trick; needs 'shaped'
    # --- who sits at the table -------------------------------------------
    scheme='solo',                       # solo | shared | duo | selfplay
    algorithm='dqn',                     # dqn | nfsp
    # --- budget -----------------------------------------------------------
    num_episodes=30000,
    evaluate_every=2500,
    num_eval_games=200,
    # --- the network and its optimiser ------------------------------------
    mlp_layers=[512, 512],
    learning_rate=0.001,
    discount_factor=1.00,
    # NFSP only. RLCard's default of 0.005 diverges the average policy here:
    # it dies between episode 5k and 17k and then returns one constant vector
    # for every state, which is uniform random play.
    sl_learning_rate=0.0001,
    # --- exploration and replay -------------------------------------------
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=50000,
    replay_memory_size=100000,
    replay_memory_init_size=100,
    update_target_estimator_every=1000,
    # --- housekeeping ------------------------------------------------------
    seed=42,
    torch_threads=1,
    verbose=False,
)

# Seats the learner fills under each scheme. Schedules below are counted in the
# agent's own steps, and an agent in m seats takes 10m steps an episode, so
# epsilon decay, replay warm-up and target refresh are all multiplied by this.
#
# 'duo' is 1, not 2. It puts *two* agents in the team seats, one each, so each
# takes ten steps an episode exactly as under 'solo'. Using 2 here would anneal
# both twice too fast and turn a cooperation comparison into an exploration one.
SEATS_PER_SCHEME = dict(solo=1, shared=2, duo=1, selfplay=4)


def build_agent(args, env, device):
    ''' Construct one learning agent

        Called twice for a 'duo'. set_seed has already seeded torch globally, so
        the second call draws a different initialisation from the advanced
        stream and the two agents genuinely differ.

    Args:
        args (Namespace): the run settings
        env (Env): the environment it will play in
        device (torch.device): where the network lives

    Returns:
        the agent
    '''
    scale = SEATS_PER_SCHEME[args.scheme]
    shared = dict(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        device=device,
        verbose=args.verbose,
    )
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        return DQNAgent(
            mlp_layers=args.mlp_layers,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=args.epsilon_decay_steps * scale,
            replay_memory_size=args.replay_memory_size,
            replay_memory_init_size=args.replay_memory_init_size * scale,
            update_target_estimator_every=args.update_target_estimator_every * scale,
            **shared,
        )
    from rlcard.agents import NFSPAgent
    return NFSPAgent(
        hidden_layers_sizes=args.mlp_layers,
        q_mlp_layers=args.mlp_layers,
        rl_learning_rate=args.learning_rate,
        sl_learning_rate=args.sl_learning_rate,
        q_discount_factor=args.discount_factor,
        q_epsilon_start=args.epsilon_start,
        q_epsilon_end=args.epsilon_end,
        q_epsilon_decay_steps=args.epsilon_decay_steps * scale,
        q_replay_memory_size=args.replay_memory_size,
        q_replay_memory_init_size=args.replay_memory_init_size * scale,
        q_update_target_estimator_every=args.update_target_estimator_every * scale,
        **shared,
    )


def seat_the_table(args, env, device):
    ''' Build the four seats and say which of them learn

        Seats 0 and 2 are one team, 1 and 3 the other.

    Args:
        args (Namespace): the run settings
        env (Env): the environment
        device (torch.device): where the networks live

    Returns:
        (tuple): the four seated agents, and {seat: learner} for the learning ones
    '''
    agent = build_agent(args, env, device)
    rule = SuecaRuleAgentV1

    if args.scheme == 'solo':          # one learner, a competent fixed partner
        return [agent, rule(), rule(), rule()], {0: agent}

    if args.scheme == 'shared':        # one network in both team seats
        return [agent, rule(), agent, rule()], {0: agent, 2: agent}

    if args.scheme == 'duo':           # two *independent* learners as a team
        partner = build_agent(args, env, device)
        return [agent, rule(), partner, rule()], {0: agent, 2: partner}

    if args.scheme == 'selfplay':      # one network in all four seats
        return [agent] * 4, {seat: agent for seat in range(4)}

    raise ValueError(f'unknown scheme {args.scheme!r}')


def train(args):
    ''' Train one run and write it to args.log_dir

    Args:
        args (Namespace): the run settings
    '''
    torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)
    device = get_device()

    env = rlcard.make(args.env, config={
        'seed': args.seed,
        'game_reward_scheme': args.reward_scheme,
        'game_reward_shaping_coefficient': args.reward_shaping_coefficient,
    })
    seats, learners = seat_the_table(args, env, device)
    env.set_agents(seats)

    # the distinct networks; under 'shared' and 'selfplay' there is only one
    distinct = list({id(a): a for a in learners.values()}.values())

    best = float('-inf')
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            for learner in distinct:
                if hasattr(learner, 'sample_episode_policy'):
                    learner.sample_episode_policy()

            trajectories, _ = env.run(is_training=True)
            # the reward of each transition is read per trick, not assumed
            # terminal, so the 'shaped' scheme can pay as the hand is played
            trajectories = reorganize_sueca(trajectories, env.get_step_rewards())

            # each seat's experience goes to the agent sitting in it. For every
            # scheme but 'duo' that is the same object; for 'duo' it is not, and
            # feeding both seats to one agent would silently make it 'shared'.
            for seat, learner in learners.items():
                for transition in trajectories[seat]:
                    learner.feed(transition)

            if episode % args.evaluate_every == 0:
                reward = tournament(env, args.num_eval_games)[0]
                logger.log_performance(episode, reward)
                # kept as a candidate, not a result: this tournament is a few
                # hundred hands and far too noisy to select on. evaluate.py
                # re-scores it on independent deals.
                if reward > best:
                    best = reward
                    save(distinct, args.log_dir, 'model_best')

    save(distinct, args.log_dir, 'model')


def save(learners, log_dir, stem):
    ''' Save a run's networks, without their replay buffers

        One learner writes <stem>.pth. A duo writes each agent separately, for
        the cross-play analysis, and the pair together under the same name so
        that evaluation needs no special case.

    Args:
        learners (list): the distinct learning agents, in seat order
        log_dir (str): the run directory
        stem (str): 'model' or 'model_best'
    '''
    with without_buffers(learners):
        if len(learners) == 1:
            torch.save(learners[0], os.path.join(log_dir, f'{stem}.pth'))
            return
        seats = [0, 2]
        for seat, learner in zip(seats, learners):
            torch.save(learner, os.path.join(log_dir, f'{stem}_seat{seat}.pth'))
        torch.save(PairAgent(dict(zip(seats, learners))),
                   os.path.join(log_dir, f'{stem}.pth'))


def train_one(settings):
    ''' Train a run in its own process, capturing its output

    Args:
        settings (dict): the run settings

    Returns:
        (tuple): the run's name, a status string, and its duration in seconds
    '''
    args = SimpleNamespace(**settings)
    started = time.perf_counter()
    os.makedirs(args.log_dir, exist_ok=True)
    console = os.path.join(args.log_dir, 'console.log')
    with open(os.path.join(args.log_dir, 'settings.txt'), 'w') as handle:
        handle.write('\n'.join(f'{k} = {v}' for k, v in sorted(settings.items())) + '\n')
    try:
        with open(console, 'w') as out, contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(out):
            train(args)
    except Exception:
        with open(console, 'a') as out:
            traceback.print_exc(file=out)
        return args.name, 'FAILED: ' + traceback.format_exc(limit=0).strip(), \
            time.perf_counter() - started
    return args.name, 'done', time.perf_counter() - started


def launch(label, runs, workers):
    ''' Train a list of runs in parallel, skipping what is already finished

    Args:
        label (str): the stage name, for the progress line
        runs (list): settings dicts, one per run
        workers (int): how many to train at a time
    '''
    pending = []
    for settings in runs:
        if os.path.exists(os.path.join(settings['log_dir'], 'model.pth')):
            continue
        if os.path.exists(settings['log_dir']):
            shutil.rmtree(settings['log_dir'])   # interrupted: restart it clean
        pending.append(settings)

    print(f'=== {label}: {len(runs) - len(pending)}/{len(runs)} already trained, '
          f'{len(pending)} to go', flush=True)
    if not pending:
        return

    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                             mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(train_one, s) for s in pending]
        for done, future in enumerate(as_completed(futures), 1):
            name, status, elapsed = future.result()
            print(f'[{done}/{len(pending)}] {name}: {status} ({elapsed/60:.1f} min)',
                  flush=True)
    print(f'=== finished in {(time.perf_counter() - started)/60:.1f} min')


# ---------------------------------------------------------------------------
# The campaign. Each stage is a function returning its runs, and nothing else:
# all of them go through launch() above.
# ---------------------------------------------------------------------------

SEEDS = (42, 43, 44)

# What stage B found, and what every later stage is built on: pay the hand trick
# by trick, discount it, and put the learner in both team seats.
BEST = dict(reward_scheme='shaped', reward_shaping_coefficient=1.00,
            discount_factor=0.90, scheme='shared')


def runs_from(root, group, conditions, seeds=SEEDS, **common):
    ''' Expand (name, overrides) pairs into one settings dict per seed

    Args:
        root (str): the stage directory under experiments/
        group (str): the folder inside it, so a group can be read on its own
        conditions (list): (name, overrides dict) pairs
        seeds (tuple): the seeds each condition is repeated under
        **common: settings shared by every run of the stage

    Returns:
        (list): settings dicts ready for launch()
    '''
    return [
        {**DEFAULTS, **common, **overrides,
         'name': f'{name}_seed{seed}',
         'seed': seed,
         'log_dir': os.path.join(ROOT, root, group, f'{name}_seed{seed}')}
        for name, overrides in conditions
        for seed in seeds
    ]


def stage_a():
    ''' The screen: one factor at a time, both algorithms, around the baseline.

        Strictly one-factor-at-a-time so every group can be read on its own. A
        full grid over eight parameters is impossible and a random search would
        confound every factor with every other.
    '''
    groups = {
        '00_baseline': [('baseline', {})],
        # what the agent is paid for - the subject of this study
        '01_reward_scheme': [
            ('victories',     dict(reward_scheme='victories')),
            ('victories_sym', dict(reward_scheme='victories_sym')),
            ('shaped0.25',    dict(reward_scheme='shaped', reward_shaping_coefficient=0.25)),
            ('shaped0.50',    dict(reward_scheme='shaped', reward_shaping_coefficient=0.50)),
            ('shaped0.75',    dict(reward_scheme='shaped', reward_shaping_coefficient=0.75)),
            ('shaped1.00',    dict(reward_scheme='shaped', reward_shaping_coefficient=1.00)),
        ],
        # a hand is ten decisions, so the baseline is undiscounted
        '02_discount_factor': [(f'gamma{g}', dict(discount_factor=g))
                               for g in (0.90, 0.95, 0.99)],
        # bracketing 1e-3 on both sides: the reward now spans [0, 120]
        '03_learning_rate': [(f'lr{lr}', dict(learning_rate=lr))
                             for lr in (0.0001, 0.0003, 0.0030, 0.0100)],
        '04_mlp_layers': [('mlp256_256', dict(mlp_layers=[256, 256])),
                          ('mlp1024_512', dict(mlp_layers=[1024, 512])),
                          ('mlp512_512_512', dict(mlp_layers=[512, 512, 512]))],
        '05_epsilon_decay': [(f'epsdecay{n}', dict(epsilon_decay_steps=n))
                             for n in (20000, 200000)],
        '06_target_update': [(f'target{n}', dict(update_target_estimator_every=n))
                             for n in (200, 5000)],
        # 'duo' is not screened here: it is the subject of its own stage, where
        # it is measured against both 'shared' and 'solo' with the cooperation
        # metrics that points alone cannot provide. See stage_coop().
        '07_scheme': [('shared', dict(scheme='shared'))],
    }
    # the algorithm is a second axis, not a condition, so the two can be read
    # against each other in every group rather than in a single row
    runs = []
    for algorithm in ('dqn', 'nfsp'):
        for group, conditions in groups.items():
            runs += runs_from('v2_stage_a', group,
                              [(f'{algorithm}_{n}', o) for n, o in conditions],
                              algorithm=algorithm)
    return runs


def stage_b():
    ''' Do the winners of the screen still help together?

        A one-factor-at-a-time screen ranks directions, not additive
        contributions. Every combination therefore ships with its own
        single-change control, in the same batch at the same length and seeds:
        without it, a combination failure cannot be told from a bad direction.
    '''
    combo = dict(reward_scheme='victories', discount_factor=0.90, scheme='shared')
    conditions = [
        ('combo', combo),
        ('combo_lr0.0001', {**combo, 'learning_rate': 0.0001}),
        ('combo_target5000', {**combo, 'update_target_estimator_every': 5000}),
        # naming a reward scheme *replaces* combo's: a run has exactly one
        ('combo_shaped1.00', {**combo, 'reward_scheme': 'shaped',
                              'reward_shaping_coefficient': 1.00}),
        ('control', dict(reward_scheme='victories')),   # the best single change alone
    ]
    return [r for algorithm in ('dqn', 'nfsp')
            for r in runs_from('v2_stage_b', '10_combination',
                               [(f'{algorithm}_{n}', o) for n, o in conditions],
                               seeds=(42, 43, 44, 45), algorithm=algorithm,
                               num_episodes=40000)]


def stage_c():
    ''' Is the agent short of time, or short of data?

        One long run, and the only stage that wants the GPU: a lone run is the
        one regime where it wins, because it is not sharing the device.
    '''
    return runs_from('v2_stage_c', '20_long', [('long300k', BEST)], seeds=(42,),
                     num_episodes=300000, evaluate_every=10000, num_eval_games=500,
                     replay_memory_size=500000, epsilon_decay_steps=300000)


def stage_d():
    ''' Many seeds of the chosen configuration, plus the two controls.

        The seeds become the ensemble members. The controls vary the replay
        buffer and the exploration length *one at a time*, which is what
        separates their effects - a buffer of 100k holds five per cent of a
        shared-seat run.
    '''
    long_run = dict(BEST, replay_memory_size=500000, epsilon_decay_steps=300000)
    common = dict(num_episodes=100000, evaluate_every=5000, num_eval_games=500)
    return (runs_from('v2_stage_d', '30_seeds', [('seeds', long_run)],
                      seeds=tuple(range(42, 54)), **common)
            + runs_from('v2_stage_d', '31_buffer',
                        [('buffer', dict(long_run, replay_memory_size=100000))], **common)
            + runs_from('v2_stage_d', '32_explore',
                        [('explore', dict(long_run, epsilon_decay_steps=50000))], **common))


def stage_nfsp():
    ''' Can NFSP be made to learn this game at all?

        Its screen measured a dead network: at RLCard's default supervised
        learning rate the average policy - the thing NFSP plays at evaluation -
        collapses to one constant vector and plays uniformly at random. These
        three groups ask whether fixing that is enough (it is not), whether the
        reward schemes rank the same for NFSP as for DQN, and whether self-play,
        the setting NFSP was designed for, rescues it.
    '''
    base = dict(algorithm='nfsp', sl_learning_rate=0.0001)
    groups = {
        '08_sl_learning_rate': [(f'sl{lr}', dict(base, sl_learning_rate=lr))
                                for lr in (0.005, 0.001, 0.0001)],
        '01_reward_scheme': [
            ('points', base),
            ('victories', dict(base, reward_scheme='victories')),
            ('victories_sym', dict(base, reward_scheme='victories_sym')),
            ('shaped0.25', dict(base, reward_scheme='shaped',
                                reward_shaping_coefficient=0.25)),
            ('shaped1.00', dict(base, reward_scheme='shaped',
                                reward_shaping_coefficient=1.00)),
        ],
        '09_nfsp_scheme': [('solo', base),
                           ('shared', dict(base, scheme='shared')),
                           ('selfplay', dict(base, scheme='selfplay'))],
    }
    return [r for group, conditions in groups.items()
            for r in runs_from('v2_stage_nfsp', group,
                               [(f'nfsp_{n}', o) for n, o in conditions])]


def stage_coop():
    ''' Do two *independent* learners cooperate?

        'shared' gets cooperation for free - its partner is itself - and 'solo'
        never has a learning partner. 'duo' puts two separate networks in the
        team seats, linked only by the shared team reward.
    '''
    return [r for scheme in ('duo', 'shared', 'solo')
            for r in runs_from('v2_coop', f'40_{scheme}',
                               [(scheme, dict(BEST, scheme=scheme))],
                               num_episodes=30000, evaluate_every=5000,
                               num_eval_games=500)]


STAGES = {'a': stage_a, 'b': stage_b, 'c': stage_c,
          'd': stage_d, 'nfsp': stage_nfsp, 'coop': stage_coop}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=sorted(STAGES), help='which stage to train')
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--dry_run', action='store_true',
                        help='list the runs without training them')
    args = parser.parse_args()

    runs = STAGES[args.stage]()
    if args.dry_run:
        for settings in runs:
            print(f'  {settings["name"]:34s} {settings["log_dir"]}')
        print(f'{len(runs)} runs')
    else:
        launch(f'stage {args.stage}', runs, args.workers)
