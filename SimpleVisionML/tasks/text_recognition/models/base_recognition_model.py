# Placeholder
PARSEQ = None
RECOGNITION_MODELS = {'parseq': PARSEQ}

class RecognitionModel:
    def __init__(self, model_info):
        self.model_info = model_info
        self.weights_path = model_info['filename']