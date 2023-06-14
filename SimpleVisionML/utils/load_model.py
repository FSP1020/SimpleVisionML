from cfg.model_dict import *

def load_detection_model(task, detection_network):
    model_dict = ModelDict()

    if task == "text":
        model_info = model_dict.text_detection_models[detection_network]
