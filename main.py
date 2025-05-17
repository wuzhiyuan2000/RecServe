# main.py
import argparse
from evaluation import evaluate_classification
from utils import load_sentiment_dataset
from recursive_serve import RecursiveServe

def main():
    parser = argparse.ArgumentParser(description="LLM Inference Service Evaluation")
    parser.add_argument(
        "--method", 
        type=str, 
        default="recserve",
        choices=["recserve"],
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="sst2", 
        choices=["sst2", "imdb", "rotten_tomatoes", "yelp_polarity", "amazon_polarity"],
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test",
    )
    args = parser.parse_args()
    
    if args.split is None:
        args.split = "test"
    
    end_model_name = "azizbarank/distilroberta-base-sst2-distilled"
    edge_model_name = "textattack/roberta-base-SST-2"
    cloud_model_name = "howey/roberta-large-sst2"
    
    dataset = load_sentiment_dataset(args.dataset, args.split)
    if args.method == "recserve":
        service = RecursiveServe(end_model_name, edge_model_name, cloud_model_name, beta=0.3, max_history_size=10000)
    else:
        raise ValueError("serving system not supported!")
    
    accuracy, report = evaluate_classification(service, dataset)
    print("\nClassification Accuracy:", accuracy)
    print("\nDetailed Report:\n", report)

if __name__ == "__main__":
    main()