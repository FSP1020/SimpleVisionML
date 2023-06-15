from .CRAFT.craft import CRAFT

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

        detection_model.eval()

        self.model = detection_model
