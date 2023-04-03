import random
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, state_, done):

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, state_, done)  # 替换存入的None
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        state, action, reward, state_, done = zip(*batch)
        return state, action, reward, state_, done

    def __len__(self):

        return len(self.buffer)


if __name__ == '__main__':
    print("")
