import torch
import torch.nn as nn

class YoloV1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YoloV1Loss, self).__init__()
        self.S = S  # Grid size (e.g., 7 for 7x7 grid)
        self.B = B  # Number of bounding boxes per grid cell
        self.C = C  # Number of classes (e.g., 20 for Pascal VOC)
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')  # Using MSE loss for all parts

    def forward(self, predictions, target):
        """
        predictions: Tensor of shape (N, S, S, B*5 + C) where
        - N: batch size
        - S: grid size (7 for 7x7 grid)
        - B: number of bounding boxes per grid cell (e.g., 2)
        - C: number of classes (e.g., 20 for Pascal VOC)
        
        target: Same shape as predictions, contains ground truth values.
        """
        N = predictions.size(0)  # Batch size

        # Reshaping predictions and target tensors
        pred_boxes = predictions[..., :self.B * 5].view(N, self.S, self.S, self.B, 5)  # Predicted boxes (x, y, w, h, conf)
        pred_classes = predictions[..., self.B * 5:]  # Predicted class probabilities

        true_boxes = target[..., :self.B * 5].view(N, self.S, self.S, self.B, 5)  # True boxes (x, y, w, h, conf)
        true_classes = target[..., self.B * 5:]  # True class labels

        # Indicator for object presence in grid cell
        obj_mask = true_boxes[..., 4]  # Confidence of the object (1 if object present, 0 otherwise)

        print("ingest data to loss completed")
        print(pred_classes.shape)
        print(true_classes.shape)
        
        ### 1. Localization Loss (coord loss) ###
        # Loss only applies to grid cells that contain an object (obj_mask = 1)
        coord_loss = self.lambda_coord * torch.sum(
            obj_mask * (
                (pred_boxes[..., 0] - true_boxes[..., 0]) ** 2 +  # x error
                (pred_boxes[..., 1] - true_boxes[..., 1]) ** 2 +  # y error
                (torch.sqrt(pred_boxes[..., 2]) - torch.sqrt(true_boxes[..., 2])) ** 2 +  # w error
                (torch.sqrt(pred_boxes[..., 3]) - torch.sqrt(true_boxes[..., 3])) ** 2    # h error
            )
        )

        ### 2. Confidence Loss ###
        # Confidence error for boxes containing an object (obj_mask = 1)
        conf_loss_obj = torch.sum(
            obj_mask * (pred_boxes[..., 4] - true_boxes[..., 4]) ** 2
        )

        # Confidence error for boxes with no object (obj_mask = 0)
        no_obj_mask = 1 - obj_mask
        conf_loss_noobj = self.lambda_noobj * torch.sum(
            no_obj_mask * (pred_boxes[..., 4] - true_boxes[..., 4]) ** 2
        )

        confidence_loss = conf_loss_obj + conf_loss_noobj

        ### 3. Classification Loss ###
        # Only for grid cells containing objects (obj_mask = 1)
        class_loss = torch.sum(
            obj_mask * torch.sum((pred_classes - true_classes) ** 2, dim=-1)
        )

        # Total loss is the sum of all three losses
        total_loss = (coord_loss + confidence_loss + class_loss) / N

        return total_loss