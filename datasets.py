import torch
import random

def make_task(vocab=1000, seq=10, samples=800):
    X = torch.randint(0, vocab, (samples, seq))
    Y = (X + random.randint(1,50)) % vocab
    return list(zip(X,Y))

def get_tasks(n=5):
    return [make_task() for _ in range(n)]
