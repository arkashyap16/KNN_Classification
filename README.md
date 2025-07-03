# 📊 K-Nearest Neighbors (KNN) Classification

This repository contains the implementation of the K-Nearest Neighbors (KNN) algorithm for classification using the Iris dataset. This task is a part of the AI & ML Internship program.

---

## 🚀 Objective

- Understand and implement KNN for classification.
- Explore the effect of different values of **K** on accuracy.
- Visualize decision boundaries using selected features.

---

## 🛠 Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📁 Dataset

- **Iris Dataset** from `sklearn.datasets`
- 3 classes: `Setosa`, `Versicolor`, `Virginica`
- 4 features: Sepal Length, Sepal Width, Petal Length, Petal Width

---

## 🔄 Workflow
1. **Load Dataset**
2. **Normalize Features** using `StandardScaler`
3. **Train-Test Split** (80%-20%)
4. **Train KNN Classifier** using `sklearn.neighbors.KNeighborsClassifier`
5. **Try K values from 1 to 20** and visualize accuracy
6. **Evaluate Final Model** using:
   - Accuracy Score
   - Confusion Matrix
7. **Visualize Decision Boundary** using first 2 features

---

## 📈 Results

- Best Accuracy Achieved at **K = 5** (may vary slightly due to train-test split)
- Confusion Matrix and decision boundaries plotted
- Normalization significantly improves model performance

---

## 📊 Output Plots

- Accuracy vs K graph
- Confusion Matrix
- Decision Boundary (2D visualization)

---
## Conclusion
- KNN is a simple yet powerful instance-based algorithm.
- Normalization is crucial for distance-based models.
- K-value selection is important and can be visualized with accuracy plots.
