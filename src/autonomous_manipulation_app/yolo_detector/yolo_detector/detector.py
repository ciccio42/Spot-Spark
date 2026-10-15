import torch
from ultralytics import YOLO
from ultralytics.utils import ThreadingLocked

class Detector:
    def __init__(self, model_path, device="cuda:0"):
        self.model_path = model_path
        self.device = device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{self.device}' but CUDA is not available")
        print(f"Initializing Detector with model path: {self.model_path}")
        self.model = self.load_model()

    def load_model(self):
        # Load the YOLO model from the specified path
        # This is a placeholder for actual model loading code
        print(f"Loading model from {self.model_path}")
        model = YOLO(self.model_path)
        model.to(self.device)
        print(f"Model loaded on {self.device}")
        return model

    @ThreadingLocked()
    def detect(self, image):
        # Perform detection on the input image using the loaded model;
        # ThreadingLocked serializes calls coming from the MultiThreadedExecutor
        print("Performing detection on the image")
        results = self.model(image, device=self.device, verbose=False)
        return results