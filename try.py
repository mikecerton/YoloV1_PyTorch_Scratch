import torch

iou_b1 = torch.tensor([[10, 40]])
iou_b2 = torch.tensor([[60, 20]])
print(iou_b1.size())
print(iou_b2)
ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
print(ious)
iou_maxes, bestbox = torch.max(ious, dim=0)
print(iou_maxes)
print(bestbox)