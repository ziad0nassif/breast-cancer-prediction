# Breast Cancer Prediction using Machine Learning

📊 **A comprehensive machine learning pipeline for breast cancer classification using multiple models**

## 📌 Overview
This project aims to classify breast cancer tumors as **Malignant (Cancerous) or Benign (Non-Cancerous)** using various **Machine Learning** algorithms. The dataset is obtained from Kaggle and preprocessed by:
- ✅ Removing highly correlated features
- ✅ Handling missing or zero values
- ✅ Normalizing feature values
- ✅ Binarization feature values
- ✅ Splitting data into training and testing sets

Several **ML models** are trained, evaluated, and compared, including:
- **Logistic Regression**  
- **K-Nearest Neighbors (KNN)**  
- **Stochastic Gradient Descent (SGD) Classifier**  
- **Random Forest**  
- **XGBoost Classifier**  

---

## 📂 Dataset
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset) and contains multiple features extracted from breast cancer cell images.

### Target Variable
- `diagnosis`: **Malignant (M) → 1, Benign (B) → 0**

### Feature Selection
Several features were removed due to high correlation, reducing redundancy and improving model performance.

---

## 🚀 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/breast-cancer-prediction.git
cd breast-cancer-prediction
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Jupyter Notebook
```bash
jupyter notebook
```
Then, open `Breast-cancer-Phase3t.ipynb` and execute the cells.

---

## 🏆 Model Performance & Results
Each model was evaluated using **accuracy** and a **confusion matrix**. Here are the results:

| **Model**                  | **Accuracy (%)**  |
|----------------------------|------------------|
| Logistic Regression        | **99.11%**       |
| K-Nearest Neighbors (KNN)  | **98.21%**       |
| SGD Classifier             | **99.11%**       |
| Random Forest Classifier   | **98.21%**       |
| XGBoost Classifier         | **99.11%**       |


**🔹 Confusion Matrices** were plotted to visualize predictions.

---

## 📊 Visualization
The notebook includes various **data visualizations** to understand feature distributions and correlations, such as:
- ✅ **Scatter Plots** of feature relationships
- ✅ **Boxplots** to identify outliers
- ✅ **Histograms** for feature distribution
- ✅ **Confusion Matrices** for model evaluation

---

## 📜 Key Takeaways
- **Feature Selection** improved performance by reducing multicollinearity.
- **Random Forest & XGBoost** provided the best accuracy.
- **Normalization & Preprocessing** were critical for model stability.

---

## 🤝 Contributions & Future Work
📌 Feel free to fork the repository and improve the model by:
- **Hyperparameter tuning** for better accuracy
- **Adding Deep Learning models** like CNNs for feature extraction
- **Optimizing feature engineering strategies**

💡 Pull requests are welcome!

---

## 📝 License
This project is open-source under the **MIT License**.
