# Recursive Offloading for LLM Serving in Multi-tier Networks  

## Introduction

RecServe is a recursive offloading framework designed for Large Language Model (LLM) serving in multi-tier networks (device-edge-cloud) , aiming to intelligently allocate LLM inference tasks across different tiers, effectively utilizing heterogeneous computational resources while reducing cross-node data transmission with minimal impact on service quality.

## Dependencies

* Python 3.x
* PyTorch
* Transformers (`transformers`)
* Scikit-learn (`scikit-learn`)
* NumPy (`numpy`)
* Datasets (`datasets`)

## Installation and Configuration

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/wuzhiyuan2000/RecServe.git
    cd RecServe
    ```

2.  **Install Dependencies**:
    It is recommended to install dependencies in a virtual environment (e.g., conda or venv).
    
    ```bash
    pip install torch transformers scikit-learn numpy datasets
    ```
    
3.  **Model Preparation**:
    The models used in the code are automatically downloaded from the Hugging Face Hub. Ensure your network connection can access Hugging Face. Default models include:
    
    * End Device: `azizbarank/distilroberta-base-sst2-distilled`
    * Edge Node: `textattack/roberta-base-SST-2`
    * Cloud: `howey/roberta-large-sst2`
    
    You can modify the model names in `main.py` or `recursive_serve.py` as needed.

## File Structure

```
RecServe/
├── main.py               # Main script for running experiments
├── recursive_serve.py    # Core implementation of the RecServe framework
├── base_serve.py         # Basic serving class
├── evaluation.py         # Evaluation metrics calculation
├── utils.py              # Utility functions for dataset loading, text cleaning, etc.
└── README.md             # This document
```

## Usage

Run evaluations using the `main.py` script.

```bash
python main.py [arguments]
```

## Bibtex Citation

```bash
@article{wu2025recursive,
  author={Wu, Zhiyuan and Sun, Sheng and Wang, Yuwei and Liu, Min and Gao, Bo and Lu, Jinda and Wu, Tingting and Yang, Zheming and Wen, Tian},
  journal={IEEE Transactions on Mobile Computing}, 
  title={Recursive Offloading for LLM Serving in Multi-tier Networks}, 
  year={2025},
  volume={},
  number={},
  pages={1-16},
  keywords={Cloud computing;Resource management;Costs;Data centers;Computational modeling;Reviews;Large language models;Computers;Computer architecture;Computational efficiency;Large language models;services computing;task offloading;edge-cloud collaboration;communication efficiency},
  doi={10.1109/TMC.2025.3642580}
}
```


