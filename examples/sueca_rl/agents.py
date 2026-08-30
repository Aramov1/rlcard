''' The two wrappers that let a team of several networks be treated as one agent.

    Both exist for the same reason: everything downstream of training - the
    evaluation protocol, the figures - places a single agent object into the
    seats it is testing. A team that is really two networks, or a committee that
    is really twelve, has nowhere to put the rest. These wrappers give it one.
'''
import contextlib

import numpy as np


class PairAgent:
    ''' Two independently trained agents playing as one team, split by seat

        Used by the 'duo' scheme, where seats 0 and 2 hold separate networks.
        The raw observation carries player_id, so the wrapper simply asks whose
        turn it is. A seat with no agent of its own falls back to the first,
        which is what lets a pair also be scored in the single-seat seating.
    '''

    use_raw = False

    def __init__(self, by_seat):
        ''' Args:
                by_seat (dict): seat id -> agent, e.g. {0: agent_a, 2: agent_b}
        '''
        self.by_seat = dict(by_seat)
        self.fallback = self.by_seat[min(self.by_seat)]

    def _for(self, state):
        return self.by_seat.get(state['raw_obs']['player_id'], self.fallback)

    def step(self, state):
        return self._for(state).step(state)

    def eval_step(self, state):
        return self._for(state).eval_step(state)

    def set_device(self, device):
        for agent in self.by_seat.values():
            if hasattr(agent, 'set_device'):
                agent.set_device(device)


class EnsembleAgent:
    ''' Several trained agents voting as one, by the mean of their Q-values

        Independently seeded runs of one configuration end up in different
        places, and averaging their opinions is the cheapest improvement
        available because the models already exist.

        The mean is taken over *masked* Q-values, so illegal actions are -inf in
        every member and stay -inf in the mean: the committee cannot vote for a
        card it may not play. Averaging assumes the members share a scale, which
        holds only while they were trained under the same reward scheme - a
        'points' member predicts values near 60 and a 'victories' member near 1,
        so mixing them is really just the 'points' member with added noise.
    '''

    use_raw = False

    def __init__(self, members):
        ''' Args:
                members (list): the trained agents to combine
        '''
        self.members = list(members)

    def eval_step(self, state):
        q = np.mean([m.predict(state) for m in self.members], axis=0)
        action = int(np.argmax(q))
        return action, {'values': q}

    def step(self, state):
        return self.eval_step(state)[0]

    def set_device(self, device):
        for member in self.members:
            if hasattr(member, 'set_device'):
                member.set_device(device)


@contextlib.contextmanager
def without_buffers(agents):
    ''' Hold the replay buffers aside while an agent is pickled

        A saved agent is only ever asked for eval_step, so its experience is
        dead weight on disk - hundreds of megabytes a run. This empties the
        buffers, yields for the save, and hands them back, so it is safe to
        call in the middle of training.

    Args:
        agents (iterable): the agents whose buffers should be set aside
    '''
    held = []
    for agent in agents:
        buffers = []
        if hasattr(agent, 'memory'):                        # dqn
            buffers.append((agent.memory, 'memory'))
        if hasattr(agent, '_rl_agent'):                     # nfsp wraps a dqn
            buffers.append((agent._rl_agent.memory, 'memory'))
        if hasattr(agent, '_reservoir_buffer'):             # nfsp supervised buffer
            buffers.append((agent._reservoir_buffer, '_data'))
        for owner, field in buffers:
            held.append((owner, field, getattr(owner, field)))
            setattr(owner, field, [])
    try:
        yield
    finally:
        for owner, field, value in held:
            setattr(owner, field, value)
