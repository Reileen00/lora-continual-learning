def forgetting(acc):
    return max(acc) - acc[-1]

acc=[0.91,0.88,0.86,0.85]
print("Forgetting:",forgetting(acc))
