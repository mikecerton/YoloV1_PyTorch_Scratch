import torch
from model import my_yoloV1
from YoloV1_loss import YoloV1Loss
from pull_Img_Label import Pull_Img_Label
torch.set_printoptions(threshold=torch.inf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

dataset = Pull_Img_Label(
    file_index='.\sample_data\sample_index.txt',
    img_dir='.\sample_data\images',
    label_dir='.\sample_data\labels',
    S=7,
    B=2,
    C=20,
    img_size=448,
)

image, label_matrix = dataset[0]

# print(image.shape)  
# print(label_matrix.shape) 
image = image.unsqueeze(0)
label_matrix = label_matrix.unsqueeze(0)

model = my_yoloV1(S = 7, B = 2, C = 20).to(device)

pred = model(image)
print(pred.shape)
# print(pred)

pred = pred.view(-1, 7, 7, 30)
print(pred.shape)
print(label_matrix.shape)

loss_obj = YoloV1Loss(S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5)
loss = loss_obj(pred, label_matrix)
print(loss)


