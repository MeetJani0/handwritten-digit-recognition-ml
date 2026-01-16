# Handwritten Digit Recognition using Classical Machine Learning

## Overview
This project implements a handwritten digit recognition system using **classical machine learning algorithms**. The goal is to classify digits (0–9) from image data and compare the performance of different models.

The project demonstrates the complete ML pipeline — data preprocessing, model implementation, evaluation, and error analysis — with a focus on interpretability and fundamentals.

---

## Models Implemented
- **K-Nearest Neighbors (KNN)** – implemented from scratch
- **Support Vector Machine (SVM)** – using scikit-learn
- **Decision Tree Classifier** – using scikit-learn

---

## Dataset
- Handwritten digit image dataset (flattened pixel values)
- Images converted into numerical feature vectors
- Standard train–test split applied

---

## Workflow
1. Data loading and exploration
2. Data preprocessing and normalization
3. Model training (KNN, SVM, Decision Tree)
4. Model evaluation and comparison
5. SVM predictions on sample test images
6. Misclassification analysis
7. Conclusion and future improvements

An execution flow diagram is included in the notebook for clarity.

---

## Evaluation Metrics
- Accuracy
- Confusion Matrix
- Classification Report

Among the evaluated models, **Support Vector Machine (SVM)** achieved the best performance due to its ability to form optimal decision boundaries in high-dimensional feature space.

---

## Model Performance

The accuracy achieved by each model on the test dataset is as follows:

| Model                     | Accuracy |
|---------------------------|----------|
| KNN (From Scratch)        | 0.9640   |
| Support Vector Machine    | 0.9852   |
| Decision Tree Classifier  | 0.8804   |

The Support Vector Machine (SVM) outperformed other models, achieving the highest accuracy due to its ability to learn optimal decision boundaries in high-dimensional feature space. The KNN model also performed well despite being implemented from scratch, while the Decision Tree showed comparatively lower generalization performance.

---

## Key Observations
- Misclassifications mainly occur between visually similar digits (e.g., 3 vs 5, 4 vs 2, 6 vs 8).
- Classical ML models struggle with spatial awareness when trained on flattened pixel values.
- Feature extraction or deep learning models could improve performance further.

---

## Future Improvements
- Apply feature extraction techniques (HOG, PCA)
- Use CNNs for spatial feature learning
- Hyperparameter tuning with GridSearchCV
- Data augmentation to handle handwriting variations

---

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## How to Run
1. Clone the repository
2. Open the Jupyter Notebook
3. Run all cells sequentially
