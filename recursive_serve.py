# recursive_serve.py
import numpy as np
from transformers import pipeline
from utils import clean_text

class RecursiveServe:
    def __init__(self, end_model_name, edge_model_name, cloud_model_name, beta=0.1, max_history_size=10000, device=0):
        self.beta = beta
        self.max_history_size = max_history_size
        self.pipelines = {
            "end": pipeline("sentiment-analysis", model=end_model_name, device=device, top_k=1),
            "edge": pipeline("sentiment-analysis", model=edge_model_name, device=device, top_k=1),
            "cloud": pipeline("sentiment-analysis", model=cloud_model_name, device=device, top_k=1)
        }
        self.confidence_history = {"end": [], "edge": [], "cloud": []}
    
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
    

    def classify_text(self, input_text, model_type="end"):
        if model_type not in self.pipelines:
            raise ValueError(f"Invalid model type: {model_type}")
        
        pipe = self.pipelines[model_type]
        prediction = pipe(input_text, truncation=True, max_length=512, top_k=1)
        predicted_label, confidence = self.process_prediction(prediction)
        self.confidence_history[model_type].append(confidence)
        if len(self.confidence_history[model_type]) > self.max_history_size:
            self.confidence_history[model_type].pop(0)
        if model_type != "cloud":
            historical = self.confidence_history[model_type]
            if len(historical) > 1:
                beta_threshold = np.percentile(historical, self.beta * 100)
                if confidence < beta_threshold:
                    next_model = "edge" if model_type == "end" else "cloud"
                    print(f"Escalating from {model_type} to {next_model} model due to low confidence ({confidence:.4f} < threshold {beta_threshold:.4f}).")
                    if model_type == "end" and next_model == "edge":
                        return self.classify_text(input_text, model_type="edge")
                    elif model_type == "edge" and next_model == "cloud":
                        return self.classify_text(input_text, model_type="cloud")
        return predicted_label, confidence

    def predict(self, input_text):
        cleaned_text = clean_text(input_text)
        return self.classify_text(cleaned_text, model_type="end")