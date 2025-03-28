## 💌 Linear Regression from Scratch

This repository contains an implementation of **Linear Regression** from scratch using **Gradient Descent**
---

## 🚀 Features
- Implements **Linear Regression** from scratch.  
- Supports **Gradient Descent** 
- Customizable **learning rate** and **number of iterations**.  
- Efficient weight initialization and training process.  



## 🛠️ Installation
Clone this repository and install the dependencies:  
```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

---

## 📚 Usage
### 1️⃣ **Import the Model**
```python
from linear_regression import LinearRegression
import numpy as np
```
### 2️⃣ **Prepare Data**
```python
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([7, 9, 11])
```
### 3️⃣ **Train the Model and prediction**
```python
model = LinearRegression(n_iterations=1000, learning_rate=0.01, no_features=2, learning_rate)
model.fit(X, y)

model.predict(X)

```




- **Gradient Descent:** Updates weights using the derivative of the cost function.  


---

## 🏆 Example Results
For input `X = [[1,2],[3,4],[5,6]]` and `y = [7, 9, 11]`, output predictions might be:  
```
Predictions: [7.1, 9.05, 10.98]
```

