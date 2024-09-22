
import torch
import torch.nn as nn
torch.set_printoptions(threshold=torch.inf)


class YoloV1_Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        """
        - Describe
            constructor of this class
        - Input
            + S (int) : number of grid cell = (S * S)
            + B (int) : number of bounding box per grid cell
            + c (int) : number of class in dataset 
            + lambda_coord (float) : lambda_coord (more info in paper)
            + lambda_noobj (float) : lambda_noobj (more info in paper)
        - Output
            None
        """
        super(YoloV1_Loss, self).__init__()
        self.S = S  
        self.B = B  
        self.C = C  
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        """
        - Describe
            calculate loss from YOLOV1 prediction and target ; N is batch_size
        - Input
            + predictions (tensor) [N, 1470] : predict answer from model
            + target (tensor) [N, 7, 7, 30] : cerrect answer from dataset
        - Output
            + total_loss (tensor) [1] : loss from preictions and target
        """
        # Reshaping predictions and target tensors
        predictions = predictions.view(-1, 7, 7, 30)
        # print(predictions)
        # print(target)
        N = predictions.size(0)  # Batch size
        # print("N = {}".format(N))
        
        pred_boxes = predictions[..., :self.B * 5].view(N, self.S, self.S, self.B, 5)  # Predicted boxes (x, y, w, h, conf)
        pred_classes = predictions[..., self.B * 5:]  # Predicted class probabilities

        # print("++  pred_box  ++")
        # print(pred_boxes)
        # print("pred_box shape = {}".format(pred_boxes.shape))

        # print("++  pred_classes  ++")
        # print(pred_classes)
        # print("pred_classes shape = {}".format(pred_classes.shape))

        true_boxes = target[..., :self.B * 5].view(N, self.S, self.S, self.B, 5)  # True boxes (x, y, w, h, conf)
        true_classes = target[..., self.B * 5:]  # True class labels

        # print("++  true_boxes  ++")
        # print(true_boxes)
        # print("true_boxes shape = {}".format(true_boxes.shape))

        # print("++  true_classes  ++")
        # print(true_classes)s
        # print("true_classes shape = {}".format(true_classes.shape))

        obj_box = true_boxes[..., 4]    # objectness of each box
        obj_grid = target[..., 4]       # objectness of each grid
        no_obj_box = 1 - obj_box        # inverst of sobjectness of each box

        # print("++  obj_box  ++")
        # print(obj_box)
        # print("obj_box shape = {}".format(obj_box.shape))

        # print("++  obj_grid  ++")
        # print(obj_grid)
        # print("obj_grid shape = {}".format(obj_grid.shape))

        # print("++  no_obj_box  ++")
        # print(no_obj_box)
        # print("no_obj_box shape = {}".format(no_obj_box.shape))

        ### 1. Localization Loss (coord loss) ###
        coord_loss = self.lambda_coord * torch.sum(
            obj_box * (
                (pred_boxes[..., 0] - true_boxes[..., 0]) ** 2 +  # x error
                (pred_boxes[..., 1] - true_boxes[..., 1]) ** 2 +  # y error
                (torch.sqrt(pred_boxes[..., 2]) - torch.sqrt(true_boxes[..., 2])) ** 2 +  # w error
                (torch.sqrt(pred_boxes[..., 3]) - torch.sqrt(true_boxes[..., 3])) ** 2    # h error
            )
        )

        # xy_loss = (pred_boxes[..., 0] - true_boxes[..., 0]) ** 2 + (pred_boxes[..., 1] - true_boxes[..., 1]) ** 2
        # wh_loss = (torch.sqrt(pred_boxes[..., 2] + 1e-6) - torch.sqrt(true_boxes[..., 2] + 1e-6)) ** 2 + (torch.sqrt(pred_boxes[..., 3] + 1e-6) - torch.sqrt(true_boxes[..., 3] + 1e-6)) ** 2

        # print("++  xy_loss  ++")
        # print(xy_loss)
        # print("xy_loss shape = {}".format(xy_loss.shape))

        # print("++  wh_loss  ++")
        # print(wh_loss)
        # print("wh_loss shape = {}".format(wh_loss.shape))

        # print("++  coord_loss  ++")
        # print(coord_loss)
        # print("coord_loss = {}".format(coord_loss.shape))

        ### 2. Confidence Loss ###
        # object case:
        conf_loss_obj = torch.sum(
            obj_box * (pred_boxes[..., 4] - true_boxes[..., 4]) ** 2
        )

        # print("++  conf_loss_obj  ++")
        # print(conf_loss_obj)
        # print("conf_loss_obj = {}".format(conf_loss_obj.shape))

        # No-object case:
        conf_loss_noobj = self.lambda_noobj * torch.sum(
            no_obj_box * (pred_boxes[..., 4] ** 2)
        )

        # print("++  conf_loss_noobj  ++")
        # print(conf_loss_noobj)
        # print("conf_loss_noobj = {}".format(conf_loss_noobj.shape))

        confidence_loss = conf_loss_obj + conf_loss_noobj

        # print("++  confidence_loss  ++")
        # print(confidence_loss)
        # print("confidence_loss = {}".format(confidence_loss.shape))
        
        ### 3. Classification Loss ###
        class_loss = torch.sum(
            obj_grid * torch.sum((pred_classes - true_classes) ** 2, dim=-1)
        )

        # print("++  class_loss  ++")
        # print(class_loss)
        # print("class_loss = {}".format(class_loss.shape))

        # Total loss is the sum of all three losses
        total_loss = (coord_loss + confidence_loss + class_loss) / N
        # print(total_loss)
        return total_loss
    
if __name__ == "__main__":
    loss = YoloV1_Loss()

    # Define the pattern for the last dimension
    pred_dim = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5] + [0.8]*20)
    target_dim = torch.tensor([0.3, 0.4, 0.5, 0.6, 1, 0.3, 0.4, 0.5, 0.6, 1] + [1]*20)

    target = target_dim.unsqueeze(0).unsqueeze(0).repeat(7, 7, 1).unsqueeze(0)
    # print(target.shape)

    tensor = pred_dim.unsqueeze(0).unsqueeze(0).repeat(7, 7, 1)
    predictions = tensor.flatten().unsqueeze(0)
    # print(tensor.shape)

    print(loss(predictions, target))
    #loss = 125.247