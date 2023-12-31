# Income-Level Prediction Project- Binary Classifiaction

## Overview

This repository has been established for a technical assessment assignment during an interview process. The goal is to deliver outcomes for a stakeholder presentation. 

This repository contains the code and documentation for a data science project focused on predicting income levels based on demographic and economic data. The project aims to address a binary classification problem, determining whether an individual's income exceeds $50,000.

## Project Description

### Steps

The following steps outline the project's progression:

0. **Exploratory Data Analysis:** Understand the dataset through visualizations and statistical analysis.
1. **Data Preprocessing:** Handle anomalies and missing values.
2. **Feature Engineering:** Optimize input variables for model training.
3. **Model Selection:** Choose suitable algorithms for binary classification.
4. **Training and Evaluation:** Train models and evaluate performance using appropriate metrics.
5. **Stakeholder Presentation:** Prepare and deliver results for a stakeholder presentation.

### Dataset

The dataset utilized in this project is modeled after the UCI/ADULT database. The target variable, representing income levels, has been binned at the $50,000 threshold. Notably, the goal field is derived from the "total person income," differing from the original ADULT database, which may influence its behavior.

### Data Source

The data originates from the United States Census Bureau, a vital source for demographic and economic information. The census is conducted every ten years, playing a crucial role in informing strategic initiatives and allocating funds for various endeavors, such as hospitals, schools, and infrastructure projects.

## Project Structure

The repository is organized as follows:

- **`data/`:** Contains the dataset used for training and evaluation.
- **`notebooks/`:** Jupyter notebooks detailing the data exploration, preprocessing, and model training processes.
- **`src/`:** Python scripts for modularized code, including data preprocessing, feature engineering, and model training.
- **`results/`:** Stores the results of the predictive models.

## Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/income-level-prediction.git


2. Create the conda environment:

   ```bash
   conda env create -f environment.yml

This will set up the required dependencies for the project.


3. Activate the conda environment:


   ```bash
   conda activate income-prediction-env


4. To run the code two approaches can be taken:

Jupyter notebooks for stakeholder presentation can be found as:
- **`notebooks/eda.ipynb`:** Jupyter notebooks detailing the data exploration.
- **`notebooks/modeling.ipynb`:** Jupyter notebooks detailing the preprocessing, and model training processes.

python scripts to run the code end to end under **`src/`:**.
- **`src/train_and_evaluate.py`:** the python script to run the train and evaluation end to end and save model as a pkl file
- **`src/predict.py`:** the python script to run the prediction end to end and save results under **`results/`:**.
   ```bash
   python src/train_and_evaluate.py
   python src/predict.py

   
Config file which contains results from the eda step can be found under **`src/config.json`:**.

### Results

## Classification Report on test set

|               | Precision | Recall  | F1-Score | Support |
|--------------:|-----------|---------|----------|---------|
|      **0**     |   0.98    |   0.92  |   0.95   |  15605  |
|      **1**     |   0.34    |   0.67  |   0.45   |   1013  |
| **Accuracy**   |           |         |   0.90   |  16618  |
| **Macro Avg**  |   0.66    |   0.79  |   0.70   |  16618  |
|**Weighted Avg**|   0.94    |   0.90  |   0.92   |  16618  |


## Classification Report on validation set:

|             | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
|      0      |   0.98    |  0.92  |   0.95   | 153,443 |
|      1      |   0.35    |  0.68  |   0.46   |  10,030 |
|-------------|-----------|--------|----------|---------|
|  **Accuracy**  |           |        |   0.90   | 163,473 |
| **Macro Avg**  |   0.66    |  0.80  |   0.70   | 163,473 |
|**Weighted Avg**|   0.94    |  0.90  |   0.92   | 163,473 |

## Conclusion

This project aims to provide a robust predictive model for income levels, contributing to informed decision-making and effective resource allocation. The structured approach involves thorough data exploration, preprocessing, and model training. The use of the United States Census Bureau data adds a significant real-world context to the predictive analysis. The results obtained from this project will be instrumental in presenting valuable insights to stakeholders and informing strategic initiatives.

For any inquiries or collaboration opportunities, please contact shima.rashidi7@gmail.com.
