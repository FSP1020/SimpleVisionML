from .CRAFT.craft import CRAFT
from tasks.text_detection.models.CRAFT.main import copyStateDict
import torch

DETECTION_MODELS = {'craft': CRAFT}

class DetectionModel:
    def __init__(self, detection_network, device, model_info):
        self.model_info = model_info
        print(model_info)
        if model_info['filename'] is not None:
            self.weights_path = model_info['filename']
        else:
            pass

        detection_model = DETECTION_MODELS[detection_network]().to(device)

        detection_model.load_state_dict(copyStateDict(torch.load(self.weights_path,  map_location='cpu')))

        detection_model.eval()

        self.model = detection_model

        self.process_file = model_info["process_file"]
