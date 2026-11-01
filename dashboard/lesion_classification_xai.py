import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO

# ==================== MODEL SETUP ====================
def load_model(weight_path):
    device = "cpu"
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7)  
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model("best_resnet_classifier.pth")
device = "cpu"

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

# ==================== GRADCAM CLASS ====================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        # Global Average Pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx

# ==================== PERTURBATION CLASS ====================
class PerturbationXAI:
    def __init__(self, model, device, num_classes=7):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.model.eval()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    def denormalize(self, tensor):
        return (tensor * self.std + self.mean).clamp(0, 1)
    
    def normalize(self, tensor):
        return (tensor - self.mean) / self.std
    
    @torch.no_grad()
    def predict(self, image_tensor):
        output = self.model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(1).item()
        confidence = probs[0, pred_class].item()
        return pred_class, confidence, probs[0]
    
    def occlusion_sensitivity(
        self, 
        image_tensor, 
        patch_size=16, 
        stride=8,
        baseline='mean',
        target_class=None
    ):
        self.model.eval()
        original_pred, original_conf, original_probs = self.predict(image_tensor)
        
        if target_class is None:
            target_class = original_pred
        
        _, _, H, W = image_tensor.shape
        sensitivity_map = np.zeros((H, W))
        count_map = np.zeros((H, W))
        
        if baseline == 'mean':
            mean_color = self.denormalize(image_tensor).mean(dim=(2, 3), keepdim=True)
            mean_color = self.normalize(mean_color)
        
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                occluded_img = image_tensor.clone()
                
                if baseline == 'mean':
                    occluded_img[:, :, i:i+patch_size, j:j+patch_size] = mean_color
                else:
                    occluded_img[:, :, i:i+patch_size, j:j+patch_size] = 0
                
                with torch.no_grad():
                    output = self.model(occluded_img)
                    probs = torch.softmax(output, dim=1)
                    occluded_conf = probs[0, target_class].item()
                
                importance = original_probs[target_class].item() - occluded_conf
                sensitivity_map[i:i+patch_size, j:j+patch_size] += importance
                count_map[i:i+patch_size, j:j+patch_size] += 1
        
        sensitivity_map = sensitivity_map / (count_map + 1e-10)
        return sensitivity_map, original_pred, original_conf
    
    def rise_explanation(
        self,
        image_tensor,
        num_masks=1000,
        mask_probability=0.5,
        mask_size=7,
        target_class=None
    ):
        self.model.eval()
        original_pred, original_conf, original_probs = self.predict(image_tensor)
        
        if target_class is None:
            target_class = original_pred
        
        _, _, H, W = image_tensor.shape
        saliency_map = torch.zeros((H, W), device=self.device)
        
        for _ in range(num_masks):
            mask = torch.rand(1, 1, mask_size, mask_size, device=self.device)
            mask = (mask < mask_probability).float()
            
            mask = torch.nn.functional.interpolate(
                mask, 
                size=(H, W), 
                mode='bilinear',
                align_corners=False
            )
            
            masked_img = image_tensor * mask
            
            with torch.no_grad():
                output = self.model(masked_img)
                probs = torch.softmax(output, dim=1)
                score = probs[0, target_class]
            
            saliency_map += mask[0, 0] * score
        
        saliency_map = saliency_map / num_masks
        return saliency_map.cpu().numpy(), original_pred, original_conf

# ==================== HELPER FUNCTIONS ====================
def overlay_heatmap(image, cam, alpha=0.4):
    """Create heatmap overlay on image"""
    # Ensure image is numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

def create_explanation_plot(original_image, cam, method_name, pred_class, confidence):
    """Create a 3-panel visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f'{method_name}\nHeatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    overlay = overlay_heatmap(original_image, cam)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add prediction info
    fig.suptitle(
        f'Prediction: {class_names[pred_class]} (Confidence: {confidence:.3f})',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)

# ==================== INITIALIZE XAI METHODS ====================
gradcam = GradCAM(model=model, target_layer=model.layer4)
perturbation_xai = PerturbationXAI(model=model, device=device, num_classes=7)

# ==================== PREDICTION FUNCTIONS ====================
def predict_basic(image):
    """Basic prediction without explanation"""
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    return {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }

def predict_with_gradcam(image):
    """Prediction with GradCAM visualization"""
    # Prepare image
    original_image = np.array(image.resize((224, 224)))
    image_tensor = transform(image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = outputs.argmax(1).item()
        confidence = probs[pred_class].item()
    
    # Generate GradCAM
    cam, _ = gradcam.generate(image_tensor, class_idx=pred_class)
    
    # Create visualization
    explanation_img = create_explanation_plot(
        original_image, cam, "GradCAM", pred_class, confidence
    )
    
    # Prepare predictions dict
    predictions = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }
    
    return predictions, explanation_img

def predict_with_occlusion(image, patch_size=16, stride=8):
    """Prediction with Occlusion Sensitivity"""
    original_image = np.array(image.resize((224, 224)))
    image_tensor = transform(image).unsqueeze(0)
    
    # Get prediction and occlusion map
    occlusion_map, pred_class, confidence = perturbation_xai.occlusion_sensitivity(
        image_tensor,
        patch_size=int(patch_size),
        stride=int(stride),
        baseline='mean'
    )
    
    # Create visualization
    explanation_img = create_explanation_plot(
        original_image, occlusion_map, "Occlusion Sensitivity", pred_class, confidence
    )
    
    # Get probabilities
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    predictions = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }
    
    return predictions, explanation_img

def predict_with_rise(image, num_masks=1000):
    """Prediction with RISE explanation"""
    original_image = np.array(image.resize((224, 224)))
    image_tensor = transform(image).unsqueeze(0)
    
    # Generate RISE map
    rise_map, pred_class, confidence = perturbation_xai.rise_explanation(
        image_tensor,
        num_masks=int(num_masks)
    )
    
    # Create visualization
    explanation_img = create_explanation_plot(
        original_image, rise_map, "RISE", pred_class, confidence
    )
    
    # Get probabilities
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    predictions = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }
    
    return predictions, explanation_img

def predict_all_methods(image):
    """Compare all explanation methods"""
    original_image = np.array(image.resize((224, 224)))
    image_tensor = transform(image).unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = outputs.argmax(1).item()
        confidence = probs[pred_class].item()
    
    # Generate all explanations
    cam, _ = gradcam.generate(image_tensor, class_idx=pred_class)
    occlusion_map, _, _ = perturbation_xai.occlusion_sensitivity(
        image_tensor, patch_size=16, stride=8
    )
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Original + Heatmaps
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    methods = [
        ("GradCAM", cam),
        ("Occlusion", occlusion_map)
    ]
    
    for idx, (name, smap) in enumerate(methods, 1):
        im = axes[0, idx].imshow(smap, cmap='jet')
        axes[0, idx].set_title(f'{name}\nHeatmap', fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')
        plt.colorbar(im, ax=axes[0, idx], fraction=0.046)
    
    # Row 2: Overlays
    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    for idx, (name, smap) in enumerate(methods, 1):
        overlay = overlay_heatmap(original_image, smap)
        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'{name}\nOverlay', fontsize=12, fontweight='bold')
        axes[1, idx].axis('off')
    
    fig.suptitle(
        f'Method Comparison - Prediction: {class_names[pred_class]} (Confidence: {confidence:.3f})',
        fontsize=16,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    comparison_img = Image.open(buf)
    
    predictions = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }
    
    return predictions, comparison_img

# ==================== GRADIO INTERFACE ====================
with gr.Blocks(title="Skin Lesion Classification with XAI") as app:
    gr.Markdown("# Skin Lesion Classification with Explainable AI")
    gr.Markdown("Upload a skin lesion image and choose an explanation method to understand the model's prediction.")
    
    with gr.Tab("Basic Prediction"):
        with gr.Row():
            with gr.Column():
                input_image_basic = gr.Image(type="pil", label="Upload Image")
                predict_btn_basic = gr.Button("Predict", variant="primary")
            with gr.Column():
                output_basic = gr.Label(num_top_classes=7, label="Predictions")
        
        predict_btn_basic.click(
            fn=predict_basic,
            inputs=input_image_basic,
            outputs=output_basic
        )
    
    with gr.Tab("GradCAM"):
        with gr.Row():
            with gr.Column():
                input_image_gradcam = gr.Image(type="pil", label="Upload Image")
                predict_btn_gradcam = gr.Button("Predict with GradCAM", variant="primary")
            with gr.Column():
                output_gradcam = gr.Label(num_top_classes=7, label="Predictions")
                explanation_gradcam = gr.Image(label="GradCAM Explanation")
        
        predict_btn_gradcam.click(
            fn=predict_with_gradcam,
            inputs=input_image_gradcam,
            outputs=[output_gradcam, explanation_gradcam]
        )
    
    with gr.Tab("Occlusion Sensitivity"):
        with gr.Row():
            with gr.Column():
                input_image_occlusion = gr.Image(type="pil", label="Upload Image")
                patch_size_slider = gr.Slider(
                    minimum=8, maximum=32, value=16, step=4,
                    label="Patch Size"
                )
                stride_slider = gr.Slider(
                    minimum=4, maximum=16, value=8, step=2,
                    label="Stride"
                )
                predict_btn_occlusion = gr.Button("Predict with Occlusion", variant="primary")
            with gr.Column():
                output_occlusion = gr.Label(num_top_classes=7, label="Predictions")
                explanation_occlusion = gr.Image(label="Occlusion Sensitivity Explanation")
        
        predict_btn_occlusion.click(
            fn=predict_with_occlusion,
            inputs=[input_image_occlusion, patch_size_slider, stride_slider],
            outputs=[output_occlusion, explanation_occlusion]
        )
    
    with gr.Tab("Compare All Methods"):
        with gr.Row():
            with gr.Column():
                input_image_compare = gr.Image(type="pil", label="Upload Image")
                predict_btn_compare = gr.Button("Compare All Methods", variant="primary")
                gr.Markdown("⚠️ This may take 20-30 seconds as it runs all methods")
            with gr.Column():
                output_compare = gr.Label(num_top_classes=7, label="Predictions")
                explanation_compare = gr.Image(label="Method Comparison")
        
        predict_btn_compare.click(
            fn=predict_all_methods,
            inputs=input_image_compare,
            outputs=[output_compare, explanation_compare]
        )
    
    gr.Markdown("""
    ### About the Methods:
    - **GradCAM**: Gradient-weighted Class Activation Mapping - highlights important regions using gradients
    - **Occlusion Sensitivity**: Tests importance by occluding patches of the image
    
    """)

if __name__ == "__main__":
    app.launch()
