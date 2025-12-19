import os
import cv2
import numpy as np

class TuckInDetector:
    def __init__(self, model_path='models/tuckin_model.h5'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                from tensorflow.keras.models import load_model as tf_load_model
                self.model = tf_load_model(self.model_path)
                print(f"✅ Tuck-in model loaded from {self.model_path}")
            except Exception as e:
                print(f"⚠️ Could not load tuck-in model at {self.model_path}: {e}. Using placeholders.")
                self.model = None
        else:
            print(f"⚠️ Model not found at {self.model_path}. Using placeholder predictions.")
            self.model = None

    def detect(self, frame):
        """Detect if shirt is tucked in or not."""
        if self.model is None:
            return np.random.choice([True, False])

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = self.model.predict(img, verbose=0)[0][0]
        return prediction < 0.5  # True if tucked in, False otherwise
