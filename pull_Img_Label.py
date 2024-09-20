import torch
from torchvision import transforms
from PIL import Image
import os

class Pull_Img_Label(torch.utils.data.Dataset):
    def __init__(self, file_index, img_dir, label_dir, S=7, B=2, C=20, img_size=448, img_transform=None):
        self.data_index = self.read_file_index(file_index)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S  # Grid size (e.g., 7 for a 7x7 grid)
        self.B = B  # Number of bounding boxes per grid cell
        self.C = C  # Number of classes
        self.matrix_size = [self.S, self.S, (self.B * 5) + self.C]  # Matrix size for target
        self.img_size = img_size  # Image size (e.g., 448x448)

        if img_transform == None:   # Any image transformations
            self.transform = transforms.Compose([transforms.Resize((448, 448)),transforms.ToTensor(),])
        else:
            self.transform = img_transform  

    def __getitem__(self, index):
        # Load image and label files
        filename = self.data_index[index]

        image_file = os.path.join(self.img_dir, filename + ".jpg")
        label_file = os.path.join(self.label_dir, filename + ".txt")

        image = Image.open(image_file).convert('RGB')  # Load image
        label_matrix = torch.zeros(self.matrix_size)  # Initialize the target label matrix

        # Open and parse the label file
        with open(label_file, 'r') as data_box:
            for box in data_box.readlines():
                data_list = box.strip().split()
                class_id, x, y, w, h = map(float, data_list)  # Get class and box details
                class_id = int(class_id)  # Class ID should be an integer

                # Convert normalized coordinates to grid cell values
                grid_x = int(x * self.S)  # Cell column where the center of the object is located
                grid_y = int(y * self.S)  # Cell row where the center of the object is located

                # Relative position of the object within the grid cell
                x_cell = x * self.S - grid_x
                y_cell = y * self.S - grid_y

                # Now that we have the grid cell, assign the bounding box and class information
                for b in range(self.B):
                    # If the confidence for the current bounding box is zero, it means it's empty
                    if label_matrix[grid_y, grid_x, b * 5 + 4] == 0:
                        label_matrix[grid_y, grid_x, b * 5 : b * 5 + 5] = torch.tensor([x_cell, y_cell, w, h, 1.0])  # Box details and confidence
                        label_matrix[grid_y, grid_x, self.B * 5 + class_id] = 1  # Class probability (one-hot)
                        break

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, label_matrix

    def __len__(self):
        return len(self.data_index)

    def read_file_index(self, file_index):
        # Read the index file which contains filenames without extensions
        with open(file_index, 'r') as myFile:
            list_data = myFile.read().strip().split("\n")
        return list_data
    
if __name__ == "__main__":

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
