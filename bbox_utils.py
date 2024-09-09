import torch

def IoU(pred_box, groundT_box, bbox_format = "yolo_format"):
    """
    - describe
        use for calculate Intersect over Union (IoU)
    - input
        + pred_box (tensor) (N * 4)  predicted bounding box from model ; N is batch_size
        + groundt_box (tensor) (N * 4) : ground truth bounding box from model ; N is batch_size
        + bbox_format (string) : choose bounding box format 1.) "yolo_format" : [cx, cy, width, height]  ; cx, cy is center of bounding box
                                                            2.) "coco_format" : [x_min, y_min, width, height]
                                                            3.) "pascal_format" : [x_min, y_min, x_max, y_max]
    - output
        + iou (tensor) (N * 1) : IoU value 
    """
    if bbox_format == "yolo_format":
        b1_x1 = pred_box[..., 0:1] - pred_box[..., 2:3] / 2
        b1_y1 = pred_box[..., 1:2] - pred_box[..., 3:4] / 2
        b1_x2 = pred_box[..., 2:3] + pred_box[..., 2:3] / 2
        b1_y2 = pred_box[..., 3:4] + pred_box[..., 3:4] / 2
        b2_x1 = groundT_box[..., 0:1] - groundT_box[..., 2:3] / 2
        b2_y1 = groundT_box[..., 1:2] - groundT_box[..., 3:4] / 2
        b2_x2 = groundT_box[..., 2:3] + groundT_box[..., 2:3] / 2
        b2_y2 = groundT_box[..., 3:4] + groundT_box[..., 3:4] / 2
    
    if bbox_format == "coco_format":
        b1_x1 = pred_box[..., 0:1]
        b1_y1 = pred_box[..., 1:2]
        b1_x2 = pred_box[..., 0:1] + pred_box[..., 2:3]  # x_min + width
        b1_y2 = pred_box[..., 1:2] + pred_box[..., 3:4]  # y_min + height

        b2_x1 = groundT_box[..., 0:1]
        b2_y1 = groundT_box[..., 1:2]
        b2_x2 = groundT_box[..., 0:1] + groundT_box[..., 2:3]  # x_min + width
        b2_y2 = groundT_box[..., 1:2] + groundT_box[..., 3:4]  # y_min + height


    if bbox_format == "pascal_format":
        b1_x1 = pred_box[..., 0:1]
        b1_y1 = pred_box[..., 1:2]
        b1_x2 = pred_box[..., 2:3]
        b1_y2 = pred_box[..., 3:4]
        b2_x1 = groundT_box[..., 0:1]
        b2_y1 = groundT_box[..., 1:2]
        b2_x2 = groundT_box[..., 2:3]
        b2_y2 = groundT_box[..., 3:4]

    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)

    intersec_area = ((x2 - x1).clamp(0)) * ((y2 - y1).clamp(0))

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = intersec_area / ((b1_area + b2_area) - intersec_area)
    
    # print(iou)
    return iou

if __name__ == "__main__":
    # pascal_format
    # a = [[100, 50, 300,200]]
    # b = [[120, 80, 310, 220]]

    # yolo_format
    # a = [[200, 125, 200, 150]]
    # b = [[215, 150, 190, 140]]

    # coco_format
    a = [[100, 50, 200, 150]]
    b = [[120, 80, 190, 140]]
    #
    x = torch.tensor(a)
    y = torch.tensor(b)
    IoU(x, y, "coco_format")