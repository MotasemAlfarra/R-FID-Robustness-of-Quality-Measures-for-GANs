import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50
import torch.nn.functional as F

class resnet(nn.Module):
    def __init__(self,
     weight_path = "/home/alfarrm/imagenet_models/smoothadv_pgd1/noise_1.0/checkpoint.pth.tar") -> None:
        super().__init__()
        model = copy_pretrained_model(
                        resnet50(pretrained=False),
                        weight_path
                        )
        self.fc_layer = model.fc
        model.fc = nn.Identity()
        self.feature_extractor = model
        

    def forward(self, x, logits=False):
        x = F.interpolate(x,
                              size=(224, 224),
                              mode='bilinear',
                              align_corners=False)

        x = self.feature_extractor(x)
        
        if logits:
            return self.fc_layer(x)
        return [x]
        # # self.model.fc = nn.Identity()
        # output.append(self.model(x))
        # return output

def copy_pretrained_model(model, path_to_copy_from):
    resnet = torch.load(path_to_copy_from, map_location='cuda')
    print(resnet.keys())
    if 'state_dict' in resnet.keys(): #For RS and ARS
        resnet = resnet['state_dict']
    if 'net' in resnet.keys(): #For MACER guys
        resnet = resnet['net']
    keys = list(resnet.keys())
    # print(keys)
    # print(resnet['fc.bias'].shape)
    count = 0
    for key in model.state_dict().keys():
        model.state_dict()[key].copy_(resnet[keys[count]].data)
        count +=1
    
    print('Pretrained model is loaded successfully')
    return model