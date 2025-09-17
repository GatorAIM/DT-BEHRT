# DT-BEHRT: Disease Trajectory-Aware Transformer for Interpretable Patient Representation Learning

## Abstract
The growing adoption of electronic health record (EHR) systems offers substantial opportunities for predictive modeling to support clinical decision making. Unlike traditional static data, structured EHRs form longitudinal trajectories of hospital visits, each consisting of diverse medical codes. Existing sequence-based, graph-based, and graph-enhanced sequence approaches aim to capture temporal dependencies and code interactions, but often treat all code types equally and overlook the central role of diagnosis codes in disease trajectories. We propose the **Disease Trajectory-aware Transformer for EHR (DT-BEHRT)**, a graph-enhanced sequence architecture that fully leverages diagnosis codes to explicitly encode ontology-guided disease interactions within organ/system categories and to model progression patterns, thereby forming clinically aligned patient representations. To strengthen learning, we design a tailored pre-training framework that combines trajectory-level code masking with ontology-guided ancestor prediction. Extensive experiments on multiple tasks show that DT-BEHRT achieves strong predictive performance while providing strong interpretability that mirrors physicians’ reasoning about disease progression.

## Model Architecture

![DT-BEHRT Architecture](figures/architecture-final.jpg)

## Requirements
The implementation of **DT-BEHRT** has been tested with the following environment:

- Python == 3.10.18  
- PyTorch == 1.13.1  
- PyTorch Geometric == 2.7.0  
- tqdm  
- scikit-learn == 1.7.0  
- scipy == 1.15.3  
- numpy == 1.26.4 