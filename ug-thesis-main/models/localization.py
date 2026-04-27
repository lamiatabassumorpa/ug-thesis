"""
GradCAM++ Localization Module
Generates forgery heatmaps showing manipulated regions.

Fix applied:
  The original alpha computation was incorrect — it used a Python list
  comprehension with one element, making alpha_den = 3*grad^2 instead of
  the correct formula: 2*grad^2 + sum_{i,j}(A_{k,i,j} * grad^3_{k,i,j}).
  The sum must aggregate over all spatial positions (H, W), not per-pixel.
"""
import torch
import numpy as np
import cv2
from typing import Optional
from pathlib import Path


class GradCAMLocalization:
    """
    GradCAM++ implementation for deepfake localization.
    Produces pixel-level forgery probability heatmaps.

    Reference: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based
    Visual Explanations for Deep Convolutional Networks", WACV 2018.
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target_layer = target_layer
        self.gradients  = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        if self.target_layer is None:
            self.target_layer = self.model.spatial_stream.backbone.conv_head

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0])
        ))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _compute_gradcampp(self) -> np.ndarray:
        """
        GradCAM++ formula (Chattopadhay et al., 2018):

            alpha^{kc}_{ij} = grad^2_{kc,ij}
                              ─────────────────────────────────────────────
                              2*grad^2_{kc,ij} + Σ_{ab}(A^{kc}_{ab} * grad^3_{kc,ab}) + ε

            w^c_k = Σ_{ij} alpha^{kc}_{ij} * ReLU(grad_{kc,ij})

            L^c_GradCAM++ = ReLU( Σ_k w^c_k * A^k )

        The key fix: Σ_{ab} is a SUM over all spatial positions (H×W),
        not a per-pixel operation. Original code applied it per-pixel.
        """
        gradients  = self.gradients.cpu().detach().numpy()[0]   # [C, H, W]
        activations = self.activations.cpu().detach().numpy()[0] # [C, H, W]

        channels, h, w = gradients.shape

        grad_sq    = gradients ** 2                         # [C, H, W]
        grad_cube  = gradients ** 3                         # [C, H, W]

        # Spatial sum of (A * grad^3) per channel → broadcast back to [C, H, W]
        spatial_sum = (activations * grad_cube).sum(axis=(1, 2), keepdims=True)  # [C, 1, 1]

        # Alpha coefficients
        alpha_num = grad_sq
        alpha_den = 2.0 * grad_sq + spatial_sum            # [C, H, W]
        alpha_den = np.where(alpha_den != 0.0, alpha_den, np.ones_like(alpha_den))
        alpha = alpha_num / (alpha_den + 1e-8)              # [C, H, W]

        # Weights: sum of alpha * ReLU(gradients) over spatial dims → [C]
        weights = (alpha * np.maximum(gradients, 0.0)).sum(axis=(1, 2))  # [C]

        # Weighted sum of activation maps
        cam = np.einsum('c,chw->hw', weights, activations)  # [H, W]
        cam = np.maximum(cam, 0.0)                          # ReLU

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.astype(np.float32)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate GradCAM++ heatmap.

        Args:
            input_tensor: [1, 3, H, W]
            target_class: 0=real, 1=fake. None → use predicted class.

        Returns:
            heatmap [H, W] in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        # Require grad for input
        input_tensor = input_tensor.requires_grad_(True)

        output, _ = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.sigmoid(output) > 0.5)

        # Binary output: output shape is [1, 1]
        target_output = output[0, 0]
        target_output.backward()

        heatmap = self._compute_gradcampp()
        return heatmap

    def visualize_heatmap(
        self,
        input_tensor: torch.Tensor,
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Returns:
            Visualization [H, W, 3] uint8 RGB
        """
        img = input_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)

        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = np.clip(img * std + mean, 0, 1)

        if heatmap.shape != img.shape[:2]:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        heatmap_u8      = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_u8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_colored = heatmap_colored.astype(np.float32) / 255.0

        vis = 0.5 * img + 0.5 * heatmap_colored
        return np.uint8(255 * np.clip(vis, 0, 1))

    def save_heatmap(
        self,
        input_tensor: torch.Tensor,
        output_path: str,
        target_class: Optional[int] = None,
        save_visualization: bool = True
    ) -> str:
        heatmap     = self.generate_heatmap(input_tensor, target_class)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if save_visualization:
            vis = self.visualize_heatmap(input_tensor, heatmap)
            cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(output_path), np.uint8(255 * heatmap))

        return str(output_path)


def generate_batch_heatmaps(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str,
    device: torch.device
) -> list:
    """Generate and save heatmaps for a batch."""
    model.eval()
    cam = GradCAMLocalization(model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for i in range(images.shape[0]):
        img   = images[i:i+1].to(device)
        label = int(labels[i].item())

        heatmap = cam.generate_heatmap(img, target_class=label)
        vis     = cam.visualize_heatmap(img, heatmap)

        with torch.no_grad():
            pred = int((torch.sigmoid(model(img)[0]) > 0.5).item())

        filename = f"sample_{i:04d}_true{label}_pred{pred}.jpg"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        saved_paths.append(str(filepath))

    cam.remove_hooks()
    return saved_paths


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING: GradCAM++ Localization Module")
    print("=" * 60)
    print("\nGradCAM++ module loaded successfully.")
    print("Note: Full test requires a trained model.")
    print("Run evaluate.py to test heatmap generation.")
