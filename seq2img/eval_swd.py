import torch
import ot

real = torch.load('feature_gt.pt').cpu()[::100]
gen  = torch.load('feature_gen.pt').cpu()[::100]

print(ot.sliced_wasserstein_distance(real, gen))