import numpy as np
import torch 
from ml.model import build_model

def predict(model, loader, device)->tuple[np.ndarray, np.ndarray]:
    """return (true_labels, class_probs) for every sample in the loader"""
    model.eval()
    all_labels=[]
    all_probs=[]

    with torch.no_grad():
        for images, labels in loader:
            outputs=model(images.to(device))
            probs=outputs.softmax(dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)

        return torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()



