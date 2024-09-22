import torch
from Loss import YoloV1Loss
import random
import torch
torch.set_printoptions(threshold=torch.inf)

loss = YoloV1Loss()

# Define the pattern for the last dimension
pred_dim = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5] + [0.8]*20)
target_dim = torch.tensor([0.3, 0.4, 0.5, 0.6, 1, 0.3, 0.4, 0.5, 0.6, 1] + [1]*20)

target = target_dim.unsqueeze(0).unsqueeze(0).repeat(7, 7, 1).unsqueeze(0)
# print(target.shape)

tensor = pred_dim.unsqueeze(0).unsqueeze(0).repeat(7, 7, 1)
predictions = tensor.flatten().unsqueeze(0)
# print(tensor.shape)

loss(predictions.repeat(2, 1), target.repeat(2, 1, 1, 1))
#loss = 125.247








