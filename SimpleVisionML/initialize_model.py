# Import basic libraries
import cv2
import numpy as np
import torch
import os
import sys

# Import other local modules (other files in this package)
from utils.load_model import *
from utils.cfg.model_dict import *

# Import logging functionality
from logging import getLogger

# Initialize logger
LOGGER = getLogger(__name__)

class SimpleModel:
    def __init__(self, use_gpu=True, device=None, model_storage_directory=None, task="text", detection=True, 
                 detection_network='craft', recognition=True, recognition_network='parseq',
                 verbose=False):
        """Create a SimpleModel

        Parameters:
            use_gpu (bool): Choose whether to use gpu (default=True)
            device (str): Choose device to run model on (default=None)

            model_storage_directory (str): Path to directory for model data. If not specified,
            models will be read from a directory as defined by the environment variable
            EASYOCR_MODULE_PATH (preferred), MODULE_PATH (if defined), or ~/.EasyOCR/.

            task (str): Choose task to complete (default=text)

            detection (bool): Choose to do detection on text (default=True)
            detection_network (str): Choose detection network to use (default=craft)

            recognition (bool): Choose to do recognition on text (default=True)
            recognition_network (str): Choose recognition network to use (defualt=parseq)

            verbose (bool): Show errors and debugging info (default=False)
        """
        self.verbose = verbose

        if use_gpu:
            # Check to see if CUDA is available
            if torch.cuda.is_available() and device not in ['cpu', 'mps']:
                self.device = 'cuda'
            # Check to see if MPS is available. MacOS user.
            elif torch.backends.mps.is_available() and device not in ['cpu', 'cuda']:
                self.device = 'mps'
            else:
                if verbose:
                    LOGGER.warning('Invalid device given. Choose from "cpu", "cuda", or "mps".\n\
                            Using CPU. Note: This module is much faster with a GPU.')
        else:
            self.device = 'cpu'

            if verbose:
                LOGGER.warning('Using CPU. Note: This module is much faster with a GPU.')
                
        # Set device index in the case that multiple GPUs are available
        self.device_index = device

        # Set model directory where model path is located
        self.model_storage_directory = model_storage_directory

        # Set model params
        self.task = task
        self.detection = detection
        self.detection_network = detection_network
        self.recognition = recognition
        self.recognition_network = recognition_network

        if detection:
            detection_model = load_detection_model(task, detection_network)

