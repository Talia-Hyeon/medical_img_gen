import torch
import numpy as np
from loss_functions.score import *

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# pred = torch.rand((2, 4, 3, 3)).to(device)
# label = torch.round(pred).to(device)
#
# metric = ArgmaxDiceScore(num_classes=5, device=device)
# dice = metric(pred, label)
# print(dice)

n_array = np.random.randint(0, 5, (2, 3))
print(n_array)
print(np.where(n_array == 1))
