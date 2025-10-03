# K-Means Clustering Project

## 📌 Objective
Perform **unsupervised learning** using **K-Means clustering** to group customers into meaningful clusters.

Dataset: [Mall Customers Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial)  
(Or synthetic dataset generated if not available.)

---

## 🛠 Tools & Libraries
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 📂 Project Steps

### 1. Load and Explore Data
```python
import pandas as pd

df = pd.read_csv("Mall_Customers.csv")
print(df.shape)
print(df.head())
