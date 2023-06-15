from cfg.model_dict import *
from tasks.text_detection.models.base_detection_model import DETECTION_MODELS
from tasks.text_recognition.models.base_recognition_model import RECOGNITION_MODELS

def load_detection_model(task, detection_network, device, verbose):
    """
    Load a detection model

    Parameters:
        task (str): Task to run detection (Ex. 'object', 'text')
        detection_network (str): Detection model/network to use

        verbose (bool): Show errors and debugging info (default=False)
    Returns:
        model (object): ML model
    """
    model_dict = ModelDict()

    # If completing text-detection
    if task == "object":
        raise NotImplementedError
    elif task == "text":
        model_info = model_dict.text_detection_models[detection_network]
    else:
        raise NotImplementedError
    
    detection_model = DETECTION_MODELS[detection_network](detection_network, model_info, device)

    return detection_model
    
    
def load_recognition_model(task, recognition_network, verbose):
    """
    Load a recognition model

    Parameters:
        task (str): Task to run recognition (Ex. 'object', 'text')
        recognition_network (str): Recognition model/network to use

        verbose (bool): Show errors and debugging info (default=False)
    Returns:
        model (object): ML model
    """
    model_dict = ModelDict()

    # If completing text-detection
    if task == "object":
        raise NotImplementedError
    elif task == "text":
        model_info = model_dict.text_recognition_models[recognition_network]
    else:
        raise NotImplementedError
    
    recognition_model = RECOGNITION_MODELS[recognition_network](model_info)

    return recognition_model
    
