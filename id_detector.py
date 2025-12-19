import os
import cv2
import numpy as np

class IDDetector:
    def __init__(self, model_path='models/id_model.h5'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                from tensorflow.keras.models import load_model as tf_load_model
                self.model = tf_load_model(self.model_path)
                print(f"✅ Model loaded from {self.model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model at {self.model_path}: {e}. Using placeholders.")
                self.model = None
        else:
            print(f"⚠️ Model not found at {self.model_path}. Detection will use placeholders.")
            self.model = None

    def detect(self, frame):
        """
        Detect if the frame contains an ID.
        Returns True if with ID, False if without ID.
        """
        if self.model is None:
            # Placeholder: random result if model not loaded
            return np.random.choice([True, False])

        # Preprocess frame for MobileNetV2
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0  # scale to [0,1]
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = self.model.predict(img, verbose=0)[0][0]
        return prediction > 0.5  # True if with_id, False if without_id
