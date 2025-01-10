import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # Scaling factor
        self.m = m  # Margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize weights and embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weights)
        
        # Add angular margin
        theta = torch.acos(torch.clamp(cosine, -1.0, 1.0))
        target_logits = torch.cos(theta + self.m)
        
        # One-hot encoding of labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Combine logits with margin
        output = one_hot * target_logits + (1.0 - one_hot) * cosine
        output *= self.s
        return output

class ArcFaceModel(nn.Module):
    def __init__(self, backbone, embedding_size, num_classes):
        super(ArcFaceModel, self).__init__()
        self.backbone = backbone  # Example: ResNet50
        self.fc = nn.Linear(backbone.out_features, embedding_size)
        self.loss = ArcFaceLoss(embedding_size, num_classes)

    def forward(self, images, labels=None):
        features = self.backbone(images)
        embeddings = self.fc(features)
        if labels is not None:
            logits = self.loss(embeddings, labels)
            return logits, embeddings
        return embeddings

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder("dataset/", transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


import torch.optim as optim

# Initialize model, optimizer, and scheduler
backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model = ArcFaceModel(backbone, embedding_size=512, num_classes=len(dataset.classes))
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(20):
    model.train()
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits, embeddings = model(images, labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    print(f"Epoch {epoch+1}/{20}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    for images, _ in validation_loader:
        images = images.to(device)
        embeddings = model(images)
        print(embeddings.shape)  # Use these for similarity comparisons


from sklearn.metrics.pairwise import cosine_similarity

def recognize_face(test_embedding, database_embeddings, database_labels):
    similarities = cosine_similarity(test_embedding, database_embeddings)
    best_match = similarities.argmax()
    return database_labels[best_match], similarities[0, best_match]


torch.save(model.state_dict(), "./models/arcface_model.pth")
print("Model saved successfully."
