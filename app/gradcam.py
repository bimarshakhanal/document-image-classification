'''
Module to generate heatmap for gradcam on the given model
'''

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import v2


transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
])
normalize = v2.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_gradcam(model, img, doc_idx):
    """
    Generates a Grad-CAM visualization for a given image and document index.

    Args:
        model: The pre-trained CNN model used for classification.
        img: The image for which to generate the Grad-CAM visualization.
        doc_idx: The index of the document class to visualize 
        (used for targeted Grad-CAM).

    Returns:
        A NumPy array representing the Grad-CAM visualization overlaid on 
        the original image.
    """

    # Preprocess the image
    # Apply necessary transformations and keep only RGB channels
    img = transform(img)[:3]
    input_tensor = normalize(img).unsqueeze(
        0)  # Normalize and add batch dimension

    # Define target layers for Grad-CAM calculation
    # Replace with the appropriate layer(s)
    target_layers = [model.model.layer4[-1]]

    # Initialize GradCAM explainer
    cam = GradCAM(model=model.model, target_layers=target_layers)

    # Define targeted Grad-CAM target based on document index
    targets = [ClassifierOutputTarget(doc_idx)]

    # Compute Grad-CAM for the input and target
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Convert to grayscale and reshape
    grayscale_cam = grayscale_cam[0, :]

    # Generate visualization by overlaying Grad-CAM on the image
    visualization = show_cam_on_image(img.permute(
        1, 2, 0).numpy(), grayscale_cam, use_rgb=True)

    return visualization
