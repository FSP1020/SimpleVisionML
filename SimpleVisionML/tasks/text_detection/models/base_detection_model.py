import torch
from collections import OrderedDict

from .CRAFT.craft import CRAFT

DETECTION_MODELS = {'craft': CRAFT}

class DetectionModel:
    def __init__(self, detection_network, device, model_info):
        self.model_info = model_info
        self.weights_path = model_info['filename']

        detection_model = DETECTION_MODELS[model_info]().to(device)

        detection_model.eval()
        
        return detection_model
