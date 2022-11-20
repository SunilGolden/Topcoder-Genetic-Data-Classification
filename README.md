# TopCoder-Genetic-Data-Classification

**Challenge Link** 

https://www.topcoder.com/challenges/75091692-5ab1-463d-9878-ace6be0508a4
<br />
<br />

# Model

**Histogram-based Gradient Boosting Classification Tree**

<br />

# Deployent Guide

### 1. First, Add the Training and Testing Data
> train.csv <br /> test_x.csv
<br />
<br />

### 2. Create Virtual Environment
```
pip install virtualenv
py -m venv env
```
<br />

### 3. Activate Virtual Environemnt

#### Example (For Windows CMD)
```
env\Scripts\activate.bat
```

#### Example (For Windows Powershell)
```
env\Scripts\Activate.psl
```

#### Example (For MAC)
```
source env\bin\activate
```
<br />

### 4. Install the Dependencies
```
pip install -r requirements.txt
```
<br />

### 5. Train the Model
```
py train.py
```
Or
```
py train.py --train_data_path='./train.csv' 
```
Step 5 will train the model and saves it as trained_model.sav
<br />
<br />

### 6. Train the Model
```
py test.py
```
Or
```
py train.py --test_data_path='./test_x.csv' --trained_model_path='./trained_model.sav' 
```
Step 6 will make predictions and saves them as solution.csv