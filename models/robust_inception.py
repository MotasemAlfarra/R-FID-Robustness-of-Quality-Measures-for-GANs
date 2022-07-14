import torch
import torch.nn as nn
from torchvision.models import inception_v3
import torch.nn.functional as F
from models.resnet import copy_pretrained_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class robust_inceptionv3(nn.Module):
    def __init__(self, path, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        super().__init__()

        # Initializing and loading the model
        model = inception_v3(pretrained=False, aux_logits=False)
        model = copy_pretrained_model(model, path)

        self.fc_layer = model.fc # Keep the FC layer for the inception score
        model.fc = nn.Identity() # Replace the FC layer with identity to get the features
        self.feature_extractor = model 

        # Normalizations used in training
        self.mean = torch.tensor(mean).view(1,3,1,1).to(device)
        self.std = torch.tensor(std).view(1,3,1,1).to(device)

    def forward(self, x, logits=False):
        x = F.interpolate(x,
                            size=(299, 299),
                            mode='bilinear',
                            align_corners=False)

        x = self.feature_extractor((x-self.mean)/self.std)
        
        if logits:
            return self.fc_layer(x)
        return [x]
