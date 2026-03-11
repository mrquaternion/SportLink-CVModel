import torch
import torchvision.transforms.v2 as transforms_v2

from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]

def infer(imageData, model, device):
    image = Image.open(imageData).convert('RGB')

    val_transform = transforms_v2.Compose([
        transforms_v2.ToImage(),
        transforms_v2.Resize(256),
        transforms_v2.CenterCrop(224),
        transforms_v2.ToDtype(torch.float32, scale=True),
    ])

    image = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)
        confidence, label_int = torch.max(probs, dim=1)

    """ idx_to_label = {
        0: 'football', 
        1: 'baseball', 
        2: 'soccer', 
        3: 'pingpong', 
        4: 'basketball', 
        5: 'petanque', 
        6: 'volleyball', 
        7: 'tennis'
    } """

    idx_to_label = {
        0: 'baseball', 
        1: 'soccer', 
        2: 'pingpong', 
        3: 'basketball', 
        4: 'petanque', 
        5: 'volleyball', 
        6: 'tennis'
    }

    return confidence.item(), idx_to_label.get(label_int.item())
