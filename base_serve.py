# base_serve.py
from transformers import pipeline

class BaseServe:
    def __init__(self, model_name, device=0):
        self.model_name = model_name
        self.pipeline = pipeline("sentiment-analysis", model=model_name, device=device, top_k=1)
        self.communication_load = {"end": 0, "edge": 0, "cloud": 0}

    def compute_confidence(self, prediction):
        return prediction[0]["score"]

    def process_prediction(self, prediction):
        predicted_label = prediction[0]["label"]
        if predicted_label.upper() in ["LABEL_1", "POSITIVE"]:
            predicted_label = "POSITIVE"
        elif predicted_label.upper() in ["LABEL_0", "NEGATIVE"]:
            predicted_label = "NEGATIVE"
        confidence = self.compute_confidence(prediction)
        return predicted_label, confidence

    def classify_text(self, input_text):
        prediction = self.pipeline(input_text, truncation=True, max_length=512, top_k=1)
        predicted_label, confidence = self.process_prediction(prediction)
        return predicted_label, confidence