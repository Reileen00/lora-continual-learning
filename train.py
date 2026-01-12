import torch, torch.nn as nn
from datasets import get_tasks
from model import LoRATransformer
from memory import Memory

model = LoRATransformer()
mem = Memory()

# freeze backbone
for p in model.tr.parameters():
    p.requires_grad=False
for p in model.emb.parameters():
    p.requires_grad=False

opt = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()),1e-3)

tasks = get_tasks()

for t,task in enumerate(tasks):
    print("Task",t)
    for x,y in task:
        x=x.unsqueeze(0); y=y.unsqueeze(0)

        replay = mem.sample(16)
        if replay:
            rx,ry = zip(*replay)
            x = torch.cat([x]+list(rx))
            y = torch.cat([y]+list(ry))

        pred = model(x)
        loss = nn.CrossEntropyLoss()(pred.view(-1,1000),y.view(-1))

        loss.backward()
        opt.step()
        opt.zero_grad()

        mem.add(x[0],y[0])
