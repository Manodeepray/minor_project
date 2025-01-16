import os
import math
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Module, Sequential
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from torch.optim import Adam
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image








# Flatten and l2_norm remain unchanged
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

##################################  MobileFaceNet #############################################################
    
class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    
    def forward(self, x):
        return self.model(x)
    
    
class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear_1 = Linear(512, embedding_size, bias=False)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        # print(x.shape)
        x = self.conv_6_dw(x)
        # print(x.shape)
    
        x = self.conv_6_flatten(x)
        # print(x.shape)
        flattened_size = x.view(1, -1).size(1)
        # print(flattened_size)
        x = self.linear_1(x)
        # print(x.shape)

        return x




############################Functions########################################################



def get_model():

    # Load the model
    model = MobileFaceNet(embedding_size=512)
    model.load_state_dict(torch.load("./models/mobilefacenet_model_trained.pth",map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    device = 'cpu'
    model.to(device)

    return model
    # Prepare the dataset or single image

def process_image(path):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode == "RGBA" else img), 
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# Load the image (replace with your own image path)
    image = Image.open(path)
    device = 'cpu'
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    return image

def truncate(embedding , i):
    
    return embedding[:,0:i]

def get_embeddings(model , image):
    
    # Get embeddings
    with torch.no_grad():
        embeddings = model(image)  # Shape: (1, embedding_size)
    print(f"Embeddings: {embeddings}")

    # Normalize embeddings for matching
    embeddings = l2_norm(embeddings)

    print(f"Embeddings Shape: {embeddings.shape}")
    print(f"Embeddings: {embeddings}")
    
    
    
    new_embeddings = truncate(embedding = embeddings , i = 100)  # Removes the batch dimension
    print(f"New Embeddings Shape: {new_embeddings.shape}")
    print(f"New Embeddings: {new_embeddings}")
    return new_embeddings



def get_database(model , directory_path):
    
    database_embeddings = {}
    
    names = os.listdir(directory_path)
    
    for name in names:
        path = os.path.join(directory_path , name)
        images = os.listdir(path)
        
        embeddings = []
        for image in images:
            img_path = os.path.join(directory_path , name , image)
            print("img_path",img_path)
            image = process_image(img_path)
            embedding = get_embeddings(model,image)
            embeddings.append(embedding)
        
        database_embeddings[name] = embeddings
        
    return database_embeddings
    
    

def recognize_face(test_embedding, database_embeddings):
    labels = [ item for item in database_embeddings]
    similarities = []
    for item in database_embeddings:
        print(item)
        embeddings = database_embeddings[item]
        # print(embeddings)
        max_sim = 0
        for embedding in embeddings:
            
            similarity = cosine_similarity(test_embedding, embedding)
            if similarity > max_sim:
                max_sim = similarity
        similarities.append(max_sim)
    
    index = similarities.index(max(similarities))
    print(f" face recognized is of {labels[index]} ")
    print(labels , similarities)
    
    
    return similarities




####################################################################################################
if __name__ == "__main__":
    
    
    model = get_model()
    image = process_image(path = "dataset/himanshu/Screenshot 2025-01-01 195907.png")
    test_embedding = get_embeddings(model , image)
    directory_path = "dataset"
    database_embeddings = get_database(model,directory_path)
    similarities = recognize_face(test_embedding, database_embeddings)
    print("similarities",similarities)
    