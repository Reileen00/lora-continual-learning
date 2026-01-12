import random

class Memory:
    def __init__(self, size=1000):
        self.buf=[]; self.size=size

    def add(self,x,y):
        if len(self.buf)>=self.size:
            self.buf.pop(0)
        self.buf.append((x,y))

    def sample(self,n):
        return random.sample(self.buf,min(n,len(self.buf)))
