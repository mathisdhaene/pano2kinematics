import cv2
import numpy as np
import torch
from torchvision import transforms


def preprocess(img: np.ndarray, device: str = "cuda") -> torch.Tensor:
    return (
        torch.from_numpy(img)
        .to(device, non_blocking=True)
        .permute(2, 0, 1)
        .float()
        .div(255)
        .unsqueeze(0)
    )


def postprocess_debug(img: torch.Tensor, to_cv2: bool = False) -> np.ndarray:
    img = img.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if to_cv2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def inspect_image_format(image):
    if isinstance(image, torch.Tensor):
        print(f"Image is already a tensor with shape: {image.shape}")
    else:
        print(f"Image is not a tensor. Type: {type(image)}")

    if isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            print(f"Image shape is valid for the pose estimator: {image.shape}")
        else:
            print(f"Unexpected tensor shape: {image.shape}")
    return image


def preprocess_for_nlf(frame):
    return (frame * 255.0).to(dtype=torch.float16, device="cuda")


def rotate_image(image, yaw):
    width = int(image.shape[1])
    pixel = int((yaw + 180) * width / 360)
    return np.roll(image, int((width / 2) - pixel), axis=1)


unNormalize = transforms.Normalize(
    mean=-np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225]),
    std=1 / np.array([0.229, 0.224, 0.225]),
)
