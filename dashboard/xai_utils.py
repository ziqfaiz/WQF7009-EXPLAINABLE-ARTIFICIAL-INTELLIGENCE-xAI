"""
Modular XAI utilities for skin lesion classification
Separate this into utils.py if you want to organize your code
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# ==================== GRADCAM ====================
class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
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
        """
        Generate GradCAM heatmap
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            class_idx: Target class (None = predicted class)
            
        Returns:
            cam: Heatmap as numpy array
            class_idx: Target class index
        """
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
    
    def __call__(self, input_tensor, class_idx=None):
        """Allow calling instance directly"""
        return self.generate(input_tensor, class_idx)


# ==================== PERTURBATION METHODS ====================
class PerturbationXAI:
    """Perturbation-based explainability methods"""
    
    def __init__(self, model, device, num_classes=7):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.model.eval()
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    def denormalize(self, tensor):
        """Denormalize image tensor"""
        return (tensor * self.std + self.mean).clamp(0, 1)
    
    def normalize(self, tensor):
        """Normalize image tensor"""
        return (tensor - self.mean) / self.std
    
    @torch.no_grad()
    def predict(self, image_tensor):
        """Get model prediction"""
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
        target_class=None,
        progress_callback=None
    ):
        """
        Occlusion Sensitivity Analysis
        
        Systematically occludes patches of the image and measures 
        the impact on the prediction confidence.
        
        Args:
            image_tensor: Input tensor [1, 3, H, W]
            patch_size: Size of occlusion patch
            stride: Stride for sliding window
            baseline: 'mean' or 'zero' for occluded region
            target_class: Class to explain (None = predicted)
            progress_callback: Optional callback(current, total)
            
        Returns:
            sensitivity_map: Importance heatmap
            pred_class: Predicted class
            confidence: Prediction confidence
        """
        self.model.eval()
        original_pred, original_conf, original_probs = self.predict(image_tensor)
        
        if target_class is None:
            target_class = original_pred
        
        _, _, H, W = image_tensor.shape
        sensitivity_map = np.zeros((H, W))
        count_map = np.zeros((H, W))
        
        # Prepare baseline
        if baseline == 'mean':
            mean_color = self.denormalize(image_tensor).mean(dim=(2, 3), keepdim=True)
            mean_color = self.normalize(mean_color)
        
        # Calculate total iterations for progress
        total_iterations = len(range(0, H - patch_size + 1, stride)) * \
                          len(range(0, W - patch_size + 1, stride))
        current_iteration = 0
        
        # Slide occlusion window
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
                
                current_iteration += 1
                if progress_callback:
                    progress_callback(current_iteration, total_iterations)
        
        # Average overlapping regions
        sensitivity_map = sensitivity_map / (count_map + 1e-10)
        return sensitivity_map, original_pred, original_conf
    
    def rise_explanation(
        self,
        image_tensor,
        num_masks=1000,
        mask_probability=0.5,
        mask_size=7,
        target_class=None,
        progress_callback=None
    ):
        """
        RISE (Randomized Input Sampling for Explanation)
        
        Generates random masks and accumulates their weighted contribution
        to the prediction to create a saliency map.
        
        Args:
            image_tensor: Input tensor [1, 3, H, W]
            num_masks: Number of random masks to generate
            mask_probability: Probability of keeping each cell
            mask_size: Size of base mask grid
            target_class: Class to explain
            progress_callback: Optional callback(current, total)
            
        Returns:
            saliency_map: Importance map
            pred_class: Predicted class
            confidence: Prediction confidence
        """
        self.model.eval()
        original_pred, original_conf, original_probs = self.predict(image_tensor)
        
        if target_class is None:
            target_class = original_pred
        
        _, _, H, W = image_tensor.shape
        saliency_map = torch.zeros((H, W), device=self.device)
        
        for mask_idx in range(num_masks):
            # Generate random mask
            mask = torch.rand(1, 1, mask_size, mask_size, device=self.device)
            mask = (mask < mask_probability).float()
            
            # Upsample to image size
            mask = torch.nn.functional.interpolate(
                mask, 
                size=(H, W), 
                mode='bilinear',
                align_corners=False
            )
            
            # Apply mask
            masked_img = image_tensor * mask
            
            # Get prediction
            with torch.no_grad():
                output = self.model(masked_img)
                probs = torch.softmax(output, dim=1)
                score = probs[0, target_class]
            
            # Accumulate weighted mask
            saliency_map += mask[0, 0] * score
            
            if progress_callback:
                progress_callback(mask_idx + 1, num_masks)
        
        # Normalize
        saliency_map = saliency_map / num_masks
        return saliency_map.cpu().numpy(), original_pred, original_conf
    
    def feature_perturbation(
        self,
        image_tensor,
        perturbation_type='gaussian_noise',
        num_samples=100,
        noise_level=0.1,
        target_class=None
    ):
        """
        Feature Perturbation Analysis
        
        Adds random perturbations and measures sensitivity.
        
        Args:
            image_tensor: Input tensor [1, 3, H, W]
            perturbation_type: 'gaussian_noise', 'salt_pepper', 'dropout'
            num_samples: Number of perturbation samples
            noise_level: Strength of perturbation
            target_class: Class to explain
            
        Returns:
            sensitivity_map: Sensitivity map
            avg_conf_drop: Average confidence drop
        """
        self.model.eval()
        original_pred, original_conf, original_probs = self.predict(image_tensor)
        
        if target_class is None:
            target_class = original_pred
        
        _, _, H, W = image_tensor.shape
        sensitivity_map = torch.zeros((H, W), device=self.device)
        confidence_drops = []
        
        for _ in range(num_samples):
            if perturbation_type == 'gaussian_noise':
                noise = torch.randn_like(image_tensor) * noise_level
                perturbed_img = image_tensor + noise
                pixel_importance = noise.abs().mean(dim=1)[0]
                
            elif perturbation_type == 'salt_pepper':
                mask = torch.rand_like(image_tensor) < noise_level
                noise = (torch.rand_like(image_tensor) > 0.5).float() * 2 - 1
                perturbed_img = image_tensor.clone()
                perturbed_img[mask] = noise[mask]
                pixel_importance = mask.float().mean(dim=1)[0]
                
            elif perturbation_type == 'dropout':
                mask = (torch.rand_like(image_tensor) > noise_level).float()
                perturbed_img = image_tensor * mask
                pixel_importance = (1 - mask).mean(dim=1)[0]
            
            with torch.no_grad():
                output = self.model(perturbed_img)
                probs = torch.softmax(output, dim=1)
                perturbed_conf = probs[0, target_class].item()
            
            conf_drop = original_probs[target_class].item() - perturbed_conf
            confidence_drops.append(conf_drop)
            sensitivity_map += pixel_importance * abs(conf_drop)
        
        sensitivity_map = sensitivity_map / num_samples
        avg_conf_drop = np.mean(confidence_drops)
        
        return sensitivity_map.cpu().numpy(), avg_conf_drop


# ==================== VISUALIZATION UTILITIES ====================
def overlay_heatmap(image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Create heatmap overlay on image
    
    Args:
        image: RGB image (H, W, 3) or PIL Image
        cam: Heatmap (h, w)
        alpha: Overlay transparency
        colormap: OpenCV colormap
        
    Returns:
        overlay: Overlaid image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Create colored heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def create_visualization(
    original_image, 
    saliency_map, 
    method_name, 
    pred_class, 
    confidence, 
    class_names,
    save_path=None
):
    """
    Create 3-panel visualization (original, heatmap, overlay)
    
    Args:
        original_image: Original image (H, W, 3)
        saliency_map: Saliency/importance map
        method_name: Name of XAI method
        pred_class: Predicted class index
        confidence: Prediction confidence
        class_names: List of class names
        save_path: Optional path to save figure
        
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(saliency_map, cmap='jet')
    axes[1].set_title(f'{method_name}\nHeatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    overlay = overlay_heatmap(original_image, saliency_map)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add title
    fig.suptitle(
        f'Prediction: {class_names[pred_class]} (Confidence: {confidence:.3f})',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_methods(
    image_tensor,
    model,
    gradcam,
    perturbation_xai,
    class_names,
    device='cpu',
    save_path=None
):
    """
    Compare multiple XAI methods side-by-side
    
    Args:
        image_tensor: Input tensor [1, 3, H, W]
        model: PyTorch model
        gradcam: GradCAM instance
        perturbation_xai: PerturbationXAI instance
        class_names: List of class names
        device: Device for computation
        save_path: Optional path to save
        
    Returns:
        fig: Matplotlib figure
    """
    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    img = (image_tensor[0] * std + mean).clamp(0, 1)
    img = img.cpu().permute(1, 2, 0).numpy()
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = outputs.argmax(1).item()
        confidence = probs[pred_class].item()
    
    # Generate explanations
    print("Generating GradCAM...")
    cam, _ = gradcam.generate(image_tensor, class_idx=pred_class)
    
    print("Generating Occlusion map...")
    occlusion_map, _, _ = perturbation_xai.occlusion_sensitivity(
        image_tensor, patch_size=16, stride=8
    )
    
    print("Generating RISE map...")
    rise_map, _, _ = perturbation_xai.rise_explanation(
        image_tensor, num_masks=800
    )
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Original + Heatmaps
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    methods = [
        ("GradCAM", cam),
        ("Occlusion", occlusion_map),
        ("RISE", rise_map)
    ]
    
    for idx, (name, smap) in enumerate(methods, 1):
        im = axes[0, idx].imshow(smap, cmap='jet')
        axes[0, idx].set_title(f'{name}\nHeatmap', fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')
        plt.colorbar(im, ax=axes[0, idx], fraction=0.046)
    
    # Row 2: Overlays
    axes[1, 0].imshow(img)
    axes[1, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    for idx, (name, smap) in enumerate(methods, 1):
        overlay = overlay_heatmap(img, smap)
        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'{name}\nOverlay', fontsize=12, fontweight='bold')
        axes[1, idx].axis('off')
    
    fig.suptitle(
        f'Method Comparison - Prediction: {class_names[pred_class]} (Confidence: {confidence:.3f})',
        fontsize=16,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("XAI utilities loaded successfully!")
    print("\nAvailable classes:")
    print("- GradCAM")
    print("- PerturbationXAI")
    print("\nAvailable functions:")
    print("- overlay_heatmap()")
    print("- create_visualization()")
    print("- compare_methods()")
