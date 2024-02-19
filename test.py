import numpy as np
import torch

tensor1 = torch.randn((3, 2, 5))
tensor2 = torch.randn((2, 2, 5))
tensor3 = torch.cat([tensor1, tensor2], dim=0)
print(tensor3.shape)
tensor11 = tensor3[3]
tensor22 = tensor3[3:]
print(tensor11.shape)
print(tensor22.shape)
