# Benchmarking Different Supervised ML-Models for Stroke Prediction

## Table of Contents

- [Project Background](#project-background)
- [Executive Summary](#executive-summary)
- [Project Framework](#project-framework)
  - [Business Understanding](#business-understanding)
  - [Data Understanding](#data-understanding)
  - [Data Preparation](#data-preparation)
  - [Modelling](#modelling)
  - [Evaluation](#evaluation)
    - [Model Performance Before Fine-Tuning](#model-performance-before-fine-tuning)
    - [Model Performance After Fine-Tuning](#model-performance-after-fine-tuning)
- [Conclusion](#conclusion)

## Project Background

This project benchmarks six supervised learning models to predict stroke, focusing on minimising false negatives while maximising precision and recall. The models evaluated include Logistic Regression, Gaussian Naïve Bayes, Decision Tree, K-Nearest Neighbours (KNN), Support Vector Machine (SVM), and Random Forest. Performance is assessed using key metrics such as training and testing accuracy, precision, recall, and the ROC curve to determine the most effective model for accurate stroke prediction.

## Executive Summary

This project follows the CRISP-DM methodology, ensuring a structured and iterative approach to stroke prediction. The dataset, sourced from Kaggle, contains 12 columns, including 8 predictive features and a binary target variable indicating stroke occurrence. Key features were pre-processed, with categorical variables encoded and numerical attributes binned to enhance interpretability. Due to a severe class imbalance, with only 4.25% of cases indicating stroke, SMOTE was applied to balance the dataset. Models were initially trained with default parameters and evaluated using accuracy, precision, recall, and ROC AUC. Based on performance, Random Forest and SVM were selected for hyperparameter tuning. After tuning, the Random Forest model emerged as the most effective, achieving 97% accuracy, 97% precision, 97% recall, and an ROC AUC of 0.99, ensuring robust classification with minimal false negatives. The SVM model improved to 95% accuracy, demonstrating strong recall and precision. Confusion matrix analysis confirmed that the Random Forest model misclassified only 49 stroke cases, while SVM misclassified 68 cases. With these results, the Random Forest model proved to be the most reliable, providing highly accurate stroke predictions while minimising false negatives.

## Project Framework

### Business Understanding

The primary goal of this project was to develop a machine learning model capable of accurately predicting stroke occurrence based on patient demographics and health indicators. The key business objective was to reduce the number of false negatives in predictions to ensure that high-risk patients are correctly identified, while maintaining high precision and recall. Understanding the implications of stroke on public health and the need for early detection shaped the overall approach to data preprocessing and model selection.

### Data Understanding

The dataset was sourced from Kaggle and contained 12 features, including demographic information such as age, gender, and residence type, as well as health-related attributes like BMI, glucose level, hypertension, and heart disease status. Initial exploration revealed a high class imbalance, with only 4.25% of instances classified as strokes, necessitating adjustments before modelling. Visualisations were used to analyse the distribution of age, glucose levels, and BMI among stroke and non-stroke patients, revealing key risk factors.

### Data Preparation

Preprocessing involved several key steps:

- **Handling Missing Values**: The dataset contained some missing BMI values, accounting for 3.66% of the data. These values were dropped, following the general rule that if missing values account for less than 10%, they can be safely removed without significantly impacting the dataset.
- **Encoding Categorical Variables**: Variables like smoking status, work type, and residence type were converted into numerical representations using one-hot encoding.
- **Feature Engineering**: Age, glucose levels, and BMI were categorised into bins to enhance model interpretability.
- **Balancing the Dataset**: Since stroke cases were severely underrepresented, SMOTE (Synthetic Minority Over-sampling Technique) was applied to generate synthetic instances of the minority class.

### Modelling

Six supervised learning models were implemented: Logistic Regression, Gaussian Naïve Bayes, Decision Tree, K-Nearest Neighbours (KNN), Support Vector Machine (SVM), and Random Forest. Models were initially trained using default hyperparameters and evaluated based on accuracy, precision, recall, and ROC AUC scores. The two best-performing models, SVM and Random Forest, were selected for hyperparameter tuning to improve classification performance.

### Evaluation

#### Model Performance Before Fine-Tuning
![ROC Curve - All Models](https://github.com/DipunMohapatra/Benchmarking-Different-Supervised-ML-Models-for-Stroke-Prediction/blob/5c01bd2203d5d219b4743ca332d8fe3f98abfac7/Visualisations/ROC%20Curve%20(All%20Models).png)
Before hyperparameter tuning, all models were trained and evaluated using accuracy, precision, recall, F1-score, and ROC AUC. The goal was to identify which models performed well in classifying stroke cases while minimising false negatives.

- **K-Nearest Neighbours (KNN)** achieved an accuracy of 93.08%, but recall was 91%, indicating that some stroke cases were missed. The precision of 92% suggests balanced classification, but further improvements were needed.
- **Logistic Regression** performed consistently across both training and test data, achieving an accuracy of 90.53%, with a precision of 90%, recall of 89%, and F1-score of 89.5%.
- **Decision Tree** was highly overfitted, with a training accuracy of 100% but a test accuracy of 93.22%. The high recall of 95% was promising, but a lower precision of 90% indicated some false positives.
- **Gaussian Naïve Bayes** underperformed, with an accuracy of 66.67%, precision of 72%, recall of 64%, and a low ROC AUC, making it unreliable for stroke prediction.
- **Support Vector Machine (SVM)** performed well, achieving an accuracy of 94.43%, with a precision of 95%, recall of 93%, and F1-score of 94%, demonstrating strong classification performance.
- **Random Forest (Before Tuning)** outperformed all models, achieving an accuracy of 96.67%, an F1-score of 97%, recall of 96%, and an ROC AUC of 0.99, confirming its robustness.

#### Model Performance After Fine-Tuning
![ROC Curve - RF After Fine-Tuning](https://github.com/DipunMohapatra/Benchmarking-Different-Supervised-ML-Models-for-Stroke-Prediction/blob/5c01bd2203d5d219b4743ca332d8fe3f98abfac7/Visualisations/ROC%20(RF%20After%20Fine%20Tuning).png)

After fine-tuning hyperparameters with RandomizedSearchCV, model performance significantly improved, particularly for the top two models: SVM and Random Forest.

- **SVM (After Tuning)** improved recall to 94%, ensuring that more stroke cases were correctly classified. The overall accuracy increased to 95%, with a precision of 96% and an F1-score of 95%, leading to fewer false negatives. The training score for SVM improved to 98.94%, while the test score reached 95.21%, demonstrating better generalisation.
- **Random Forest (After Tuning)** remained the top-performing model, improving recall to 97%, maintaining an accuracy of 97%, and achieving an F1-score of 97%. The ROC AUC remained at 0.99, confirming its superior predictive performance.

The reduction in false negatives through fine-tuning is crucial in medical applications, as it ensures that high-risk individuals are accurately identified for preventive intervention.

## Conclusion

This project successfully developed and evaluated machine learning models to predict stroke occurrence, following the CRISP-DM framework. The dataset was preprocessed to address missing values, encode categorical variables, and balance class distribution using SMOTE. Six models were trained and assessed, with Random Forest and SVM emerging as the top-performing models. Fine-tuning further enhanced Random Forest’s recall and accuracy, making it the most reliable model for stroke prediction.
