

def watch(mem=2000):
    pass


import torch

t = torch.cuda.get_device_properties(0).total_memory
c = torch.cuda.memory_cached(0)
a = torch.cuda.memory_allocated(0)
print(torch.cuda.max_memory_allocated(0))


print(torch.cuda.get_device_properties(0))