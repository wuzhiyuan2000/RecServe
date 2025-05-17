# evaluation.py
from sklearn.metrics import accuracy_score, classification_report

def evaluate_classification(service, dataset):
    true_labels = []
    predicted_labels = []
    
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    
    print(dataset)
    
    for i, example in enumerate(dataset):
        text = example["text"]
        label = example["label"]
        label_string = label_map[label]
        result = service.predict(text)
        if isinstance(result, list):
            best = max(result, key=lambda x: x[2])
            predicted_label, confidence = best[1], best[2]
        else:
            predicted_label, confidence = result
        
        true_labels.append(label_string)
        predicted_labels.append(predicted_label)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} examples...")
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(
        true_labels, predicted_labels, target_names=["NEGATIVE", "POSITIVE"]
    )
    return accuracy, report