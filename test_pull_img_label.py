import torch
from pull_Img_Label import Pull_Img_Label
torch.set_printoptions(threshold=torch.inf)

# obj = Pull_Img_Label(".\sample_data\sample_index.txt", ".\sample_data\images", ".\sample_data\labels")

# obj.__getitem__(0)

# Create a dataset with transformations
dataset = Pull_Img_Label(
    file_index='.\sample_data\sample_index.txt',
    img_dir='.\sample_data\images',
    label_dir='.\sample_data\labels',
    S=7,
    B=2,
    C=20,
    img_size=448,
)

# Get the first image and label
image, label_matrix = dataset[0]

print(image.shape)  # Should be torch.Size([3, 448, 448])
print(label_matrix.shape)  # Should be torch.Size([7, 7, 30])

print(label_matrix)