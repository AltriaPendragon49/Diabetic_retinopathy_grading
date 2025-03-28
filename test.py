import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from train import RetinopathyModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def predict(model, image_path, transform):
    device = torch.device("cuda")
    model = model.to(device) 
    model.eval() 
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)#增加批次维度

    with torch.no_grad():  
        outputs = model(image) 
        _, predicted = torch.max(outputs.data, 1)
        
    return predicted.item()

def main():
    model = RetinopathyModel()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'model', 'final_model.pth')))
    
    transform = transforms.Compose([#图像处理
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    

    test_dir = os.path.join(BASE_DIR, 'test')  
    predictions = []  
    image_ids = []  
    
    for image_name in sorted(os.listdir(test_dir), key=lambda x: int(x.split('.')[0])):#按图像ID排序
        if image_name.endswith('.jpg'): 
            image_path = os.path.join(test_dir, image_name)  
            image_id = image_name.split('.')[0]#获取id
            
            prediction = predict(model, image_path, transform)#进行预测
            
            image_ids.append(image_id)
            predictions.append(prediction)  

    submission = pd.DataFrame({
        'Image': image_ids,  
        'Predict': predictions  
    })
    
    submission.to_csv(os.path.join(BASE_DIR, 'submission.csv'), index=False)  
    print("预测完成")  

if __name__ == "__main__":
    main()