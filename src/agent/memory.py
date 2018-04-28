import numpy as np
from collections import deque
import random
import itertools


class RingMemory:
    def __init__(self, max_size, cols=5):
        self.data = np.zeros([max_size, cols], dtype=object)
        self.max_size = max_size
        self.size = 0

    def append(self, item):
        if self.size >= self.max_size: 
            self.data[0] = item
            self.data = np.roll(self.data, -1, axis=0)
        else:
            self.data[self.size] = item
            self.size += 1

    def sample(self, size):
        idx = np.random.choice(self.size, size=size, replace=False)
        return np.copy(self.data[idx,:])


class EpisodeMemory():
    def __init__(self, max_size, cols=5):
        self.data = deque()
        self.max_size = max_size
        self.size = 0
        self.cols = cols

    def append(self, item):
        self.data.append(item)
        if self.size >= self.max_size: 
            self.data.popleft()
        else:
            self.size += 1

    def sample(self, batch_size, trace_size):
        population = list(self.data)
        assert len(population) > batch_size
        episodes = random.sample(population, batch_size)
        pnt = [random.randint(0,len(ep)-trace_size) for ep in episodes]
        traces = [t for i, ep in enumerate(episodes) 
                                for t in ep[pnt[i]:pnt[i]+trace_size]]
        traces = np.array(traces)
        return traces





