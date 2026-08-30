# Sueca RL

Everything needed to reproduce the Sueca experiments, in four files.

| file | what it is |
|---|---|
| `train.py` | the training engine, and every stage of the campaign as a configuration |
| `evaluate.py` | the scoring protocol, and the ensemble builder |
| `agents.py` | `PairAgent` (a two-network team) and `EnsembleAgent` (a committee) |
| `cooperation.py` | the assist rate and cross-play, for the cooperation stage |

## Install

```bash
pip3 install -e .
pip3 install -e .[torch]
```

## The two commands

Everything is *train a stage*, then *score it*. Runs land in
`experiments/<stage>/<group>/<run>/`, and scoring reads those directories, so no
figure ever needs retraining.

```bash
cd <repo root>
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1        # see "Run on the CPU" below

python examples/sueca_rlv2/train.py a --workers 14    # train
python examples/sueca_rlv2/evaluate.py experiments/v2_stage_a    # score
```

`train.py <stage> --dry_run` lists what a stage would train without training it.
Every stage is resumable: a run that already holds `model.pth` is skipped, so an
interrupted launch is continued by repeating the same command.

## How each result is obtained

| result | command | runs | approx. time |
|---|---|---|---|
| Stage A, the screen | `train.py a --workers 14` | 132 | ~20 h |
| Stage B, do the winners combine | `train.py b --workers 8` | 40 | ~10 h |
| Stage C, does longer help | `CUDA_VISIBLE_DEVICES=0 train.py c` | 1 | ~8 h |
| Stage D, seeds + buffer/exploration | `train.py d --workers 9` | 18 | ~14 h |
| The NFSP block | `train.py nfsp --workers 14` | 33 | ~11 h |
| Stage E, cooperation | `train.py coop --workers 9` | 9 | ~1 h |

Times are for a 16-core laptop. NFSP runs cost roughly four times a DQN run at
equal episodes, because it trains a supervised average policy and a reservoir on
top of its best response.

Score any of them with `evaluate.py experiments/<stage>`. The final agent is an
ensemble of the stage D seeds:

```bash
python examples/sueca_rlv2/evaluate.py --ensemble experiments/v2_final/ens12 \
    --models 'experiments/v2_stage_d/30_seeds/*/model_best.pth'
python examples/sueca_rlv2/evaluate.py experiments/v2_final --num_games 2500 \
    --seeds 0 1 2 3 4 5 6 7
```

The cooperation stage has two measurements that points alone cannot give:

```bash
python examples/sueca_rlv2/cooperation.py experiments/v2_coop
```

## Checking you reproduced it

Two reference lines appear in every table and are the fastest sanity check. They
are properties of the protocol, not of any agent, so they should come out the
same for you:

```
rule agent (parity)     60.0 points, seating-independent, ~49% of hands won
random agent (floor)    55.6 solo / 50.5 team
```

The rule agent's score is seating-independent because putting *it* under test
fills all four seats with rule agents either way. Its expected value is exactly
60.0 — half the deck — so whatever it actually measures is the protocol's
residual noise. If those two lines are right, the pipeline is right.

For reference, the headline result: twelve stage D seeds ensembled from their
best checkpoints score **63.3 points as a team against the rule player's 60.0**,
winning 56.6% of hands against 49.0%, over 20,000 paired hands.

## Two things that are easy to get wrong

**Run the sweeps on the CPU.** The networks are small, so a run is bound by
Python game logic and per-batch kernel launches, not by matrix multiplication.
One run is faster on the GPU — 47 episodes/s against 31 on a core — but
processes serialise on the device, so twelve concurrent runs give 89 episodes/s
in total on the GPU against 115 on the CPU. Only stage C, a single long run,
wants the GPU.

**Schedules are counted in the agent's own steps, not in episodes.** An agent
occupying *m* seats takes 10*m* steps per episode, so `SEATS_PER_SCHEME`
stretches ε-decay, replay warm-up and target refresh by *m*:

```
solo 1     one learner, a fixed rule partner
shared 2   one network in both team seats - it takes twenty steps an episode
duo 1      TWO networks, one per team seat - each takes ten, like solo
selfplay 4 one network in all four seats
```

`duo` is 1, not 2. Using 2 there would anneal both agents twice too fast and
quietly turn a cooperation comparison into an exploration comparison.

## Notes

Seeds are environment-local: `Env.seed` creates `self.np_random` and injects it
into the game, and game code draws only from it, so parallel runs never
interfere. Evaluation re-seeds before each agent, so every agent meets an
identical sequence of deals and comparisons within a batch are exactly paired.

A run keeps two models: `model.pth`, the final weights, and `model_best.pth`,
the best by the run's own periodic tournament. The second is a *candidate*, not
a result — it is selected on a few hundred hands, far too few to trust, and
`evaluate.py` re-scores it on independent deals before it is believed. A `duo`
run also writes `model_seat0.pth` and `model_seat2.pth`, which is what
`cooperation.py` re-pairs for cross-play.

A 500k replay buffer costs about 2 GB per run, so stage D is capped at nine
concurrent workers on a 30 GB machine.
