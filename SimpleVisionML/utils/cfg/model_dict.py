from tasks.text_detection.models.CRAFT.main import processFile

class ModelDict:
    def __init__(self):
        self.text_detection_models = {"craft": {'filename': 'SimpleVisionML/tasks/text_detection/models/CRAFT/weights/craft_mlt_25k.pth', 
                                                'process_file': processFile}}
        self.text_recognition_models = {"parseq": {'filename': 'not_implemented'}}