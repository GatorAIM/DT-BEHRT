# DT-BEHRT: Disease Trajectory-Aware Transformer for Interpretable Patient Representation Learning

## Abstract
The growing adoption of electronic health record (EHR) systems has provided unprecedented opportunities for predictive modeling to guide clinical decision making. Structured EHRs contain longitudinal observations of patients across hospital visits, where each visit is represented by a set of medical codes. While sequence-based, graph-based, and graph-enhanced sequence approaches have been developed to capture rich code interactions over time or within the same visits, they often overlook the inherent heterogeneous roles of medical codes arising from distinct clinical characteristics and contexts. To this end, in this study we propose the Disease Trajectory-aware Transformer for EHR (DT-BEHRT), a graph-enhanced sequential architecture that disentangles disease trajectories by explicitly modeling diagnosis-centric interactions within organ systems and capturing asynchronous progression patterns. To further enhance the representation robustness, we design a tailored pre-training methodology that combines trajectory-level code masking with ontology-informed ancestor prediction, promoting semantic alignment across multiple modeling modules. Extensive experiments on multiple benchmark datasets demonstrate that DT-BEHRT achieves strong predictive performance and provides interpretable patient representations that align with clinicians’ disease-centered reasoning.

## Model Architecture

![DT-BEHRT Architecture](figures/architecture-final.jpg)

## Requirements
The implementation of **DT-BEHRT** has been tested with the following environment:

- Python==3.10.18  
- PyTorch==1.13.1  
- torch_geometric==2.7.0  
- tqdm  
- icd-mappings  
- scikit-learn==1.7.0  
- scipy==1.15.3  
- numpy==1.26.4 
- pandas==2.3.1  

## Replicating the Results
1. **Download the raw data**  
   From the [MIMIC-III dataset](https://physionet.org/content/mimiciii/1.4/), obtain the following files (**no need to unzip**):  
   - `ADMISSIONS.csv.gz`  
   - `DIAGNOSES_ICD.csv.gz`  
   - `LABEVENTS.csv.gz`  
   - `PATIENTS.csv.gz`  
   - `PRESCRIPTIONS.csv.gz`  
   - `PROCEDURES_ICD.csv.gz`  

   Similarly, download the corresponding files from the [MIMIC-IV dataset](https://physionet.org/content/mimiciv/3.1/) (located in the `hosp` folder).  
   Place the files into the following directories:  
   - `./data_process/MIMIC-III-raw`  
   - `./data_process/MIMIC-IV-raw`  

2. **Preprocess the data**  
   Run the notebooks:  
   - `MIMIC-III.ipynb`  
   - `MIMIC-IV.ipynb`  
   
   These notebooks will perform the necessary preprocessing steps and prepare the datasets for modeling.  

3. **Test the model**  
   Run the notebook **`test_model.ipynb`**.  
   - To evaluate performance on different tasks, adjust the `task_index` in the configuration:  
     - `0`: In-hospital mortality  
     - `1`: Readmission  
     - `2`: Prolonged length of stay (PLOS)  
 