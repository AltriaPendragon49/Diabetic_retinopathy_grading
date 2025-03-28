import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class RetinopathyDataset(Dataset):#数据预处理
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)  
        self.img_dir = img_dir  
        self.transform = transform
        
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.data.iloc[idx, 0]}.jpg")#图像名称
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]#病变等级
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class RetinopathyModel(nn.Module):#模型
    def __init__(self):
        super(RetinopathyModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features #获取全连接层的输入特征数
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)#正常或病变
        )
        
    def forward(self, x):
        return self.resnet(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" )  
    model = model.to(device)
    
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)#学习率调度器  
    best_val_loss = float('inf')#最佳验证损失初始无穷大
    
    for epoch in range(num_epochs):#训练模型
        model.train()  
        running_loss = 0.0  
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)  
            labels = labels.to(device)  
            
            optimizer.zero_grad()  
            outputs = model(inputs) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()#更新参数
            
            running_loss += loss.item() 
        scheduler.step()#更新学习率
        
        model.eval()#验证  
        val_loss = 0.0 
        correct = 0  
        total = 0  
        
        with torch.no_grad():  
            for inputs, labels in val_loader:
                inputs = inputs.to(device)  
                labels = labels.to(device)  
                outputs = model(inputs)  
                loss = criterion(outputs, labels)  
                val_loss += loss.item()  
                
                _, predicted = torch.max(outputs.data, 1)#每行的预测结果
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_val_loss = val_loss / len(val_loader)  
        accuracy = 100 * correct / total 
        
        print(f'轮数：[{epoch+1}/{num_epochs}]')
        print(f'准确率：{accuracy:.2f}%')  
        
        if epoch_val_loss < best_val_loss:#如果当前验证损失更低
            torch.save(model.state_dict(), os.path.join(BASE_DIR, 'model','final_model.pth')) 

def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),#水平
        transforms.RandomVerticalFlip(),#垂直
        transforms.RandomRotation(20),#旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset = RetinopathyDataset(
        csv_file=os.path.join(BASE_DIR, 'Mess1_annotation_train.csv'), 
        img_dir=os.path.join(BASE_DIR, 'train'),
        transform=transform 
    )
    
    train_size = int(0.8 * len(dataset))  
    val_size = len(dataset) - train_size 
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])#随机划分数据集
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  
    
    model = RetinopathyModel()  
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30) 
    
if __name__ == "__main__":
    train() 