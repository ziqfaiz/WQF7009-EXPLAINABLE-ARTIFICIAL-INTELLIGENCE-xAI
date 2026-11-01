import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def load_model(weight_path):
    device = "cpu"

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7)  

    state_dict = torch.load("best_resnet_classifier.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model("best_resnet_classifier.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = [
    'Melanoma',
    'Melanocytic nevus', 
    'Basal cell carcinoma',
    'Actinic keratosis',
    'Benign keratosis',
    'Dermatofibroma',
    'Vascular lesion'
]

def predict(image):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    return {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }


import gradio as gr

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=7),
    title="Skin Lesion Classification",
    description="Upload a skin lesion image for classification"
)

app.launch()
