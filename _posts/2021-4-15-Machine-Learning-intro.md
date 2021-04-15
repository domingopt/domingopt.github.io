---
layout: post
title: Introduction to Machine Learning
mathjax: true
categories:
  - Machine Learning
tags:
  - beginner
  - supervised learning
  - classification
  - regression
---

In this post the main objectives for Machine Learning and some of the key concepts will be introduced. You will learn:
* How is the Machine Learning paradigm different from other programming approaches.
* Difference of classification vs prediction problems.
* How to define a machine learning problem and how to split the data available between training and testing.
* How to compare different models.
* Underfitting and overfitting and the balance between bias and variance.

## How is Machine Learning different from traditional algorithms?

In a traditional algorithm, given a set of inputs, a computer program will follow a set of (in general deterministic) steps to reach out a solution. So in more formal terms, given an $${x}$$, a vector of inputs, we get to our best estimate $$\hat{y}$$ of the ground truth $y$ by applying the function $$f$$:

> $$f(x) = \hat{y}$$

where $f$ represents the algorithm that we have implemented.

In traditional supervised learning cases, the goal is to *learn* $f()$ given enough observations of both ${x}$ and $y$. This difference is very important, so let's take a second to ponder on that. In traditional algorithms, you write $f()$ so once you are given $x$, you produce $\hat{y}$ (which is your estimation of $y$). In Statistical Leaning, you are given a set of $x$ **and $y$** so you produce $f()$ and when given new data ${x}\prime$ you can generate $\hat{y}$.

This might sound all a bit theoretical, so lets try to make it more concrete. Imagine you get tasked by a scientist to classify flowers in 3 categories, based on certain inputs (or in the ML lingo **features**). He has even collected data from his past analysis and would like to automate this classification.

Using traditional algorithms, one would end up with a series of `if then else` statements to end up classifying the flowers based on those inputs. While this approach might work when we manage to uncover the relationships between these features and the classes and when the number of *features* is not too large, it becomes very quickly difficult to implement when the relationships between the features and the output are not that clear and when the number of features increases.

In contrast, with a Statistical Learning algorithm, we will feed it with a battery of examples of inputs and outputs and eventually (and hopefully!) the algorithm learn the importance of these factors and how to combine them to get to the correct answer.

## Classification vs Prediction

Two of the biggest families of problems in the supervised learning world are *classification* and *prediction* ones (although the purists will argue that there is no classification, as a classification problem is a prediction one, where the prediction universe is a *categorical* one).

Again, let's go for an example. If you are trying to separate data into different groups or categories, you are dealing with a classification problem (for example, given some flower details decide what type of flower it is, given an email classify it as spam or non-spam or given a picture, classify it as a dog or a cat).

On the other hand, if your aim is to anticipate client demand or to predict the price of a house given certain information, you are dealing with a prediction problem. It is important to understand this as the family of algorithms to model those that you might end up using will be different.


```python
import numpy as np
import pandas as pd
from sklearn import datasets
from plotnine import *
import itertools
```


```python
iris_data = datasets.load_iris()
iris_features, iris_classes = iris_data['data'], iris_data['target']
iris_df = pd.DataFrame(iris_features)
iris_df['classification'] = iris_classes
iris_df.rename({0: 's_lenght', 1: 's_width', 2: 'p_length', 3: 'p_width'}, inplace=True, axis=1)
iris_df['classification'] = iris_df['classification'].astype('category')
```


```python
for axis1, axis2 in itertools.combinations(iris_df.columns[:-1], 2):
  display(ggplot(iris_df, aes(x=axis1, y=axis2, fill='classification')) + geom_point())
```

    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):



    
<img src="/images/MachineLearning_intro_files/MachineLearning_intro_5_1.png">
    



    <ggplot: (8754541070825)>


    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):



    
<img src="/images/MachineLearning_intro_files/MachineLearning_intro_5_4.png">
    



    <ggplot: (8754541116105)>


    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):



    
<img src="/images/MachineLearning_intro_files/MachineLearning_intro_5_7.png">
    



    <ggplot: (8754539439137)>


    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):



    
<img src="/images/MachineLearning_intro_files/MachineLearning_intro_5_10.png">
    



    <ggplot: (8754539459597)>


    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):



    
<img src="/images/MachineLearning_intro_files/MachineLearning_intro_5_13.png">
    



    <ggplot: (8754539782173)>


    /usr/local/lib/python3.7/dist-packages/plotnine/utils.py:1246: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
      if pdtypes.is_categorical(arr):



    
<img src="/images/MachineLearning_intro_files/MachineLearning_intro_5_16.png">
    



    <ggplot: (8754539327677)>



```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

```


```python
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop('classification', 1), iris_df['classification'], test_size=0.4, random_state=42)
```


```python
classifiers = {'K-Nearest Neighbors': KNeighborsClassifier(3),
               'Logistic Regression': LogisticRegression(),
               'SVM Linear kernel': SVC(kernel="linear", C=0.025),
               'SVM': SVC(gamma=2, C=1),
               'Decission Tree': DecisionTreeClassifier(max_depth=5),
               'Random Forest': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               'AdaBoost': AdaBoostClassifier(),
               'GBM': GradientBoostingClassifier(n_estimators=10),
               'XGBoost': XGBClassifier(n_estimators=10)}

results = {'model':[], 'accuracy_train':[], 'accuracy_test':[]}

for name, classifier in classifiers.items():
  classifier.fit(X_train, y_train)
  train_score = classifier.score(X_train, y_train)
  test_score = classifier.score(X_test, y_test)
  results['model'].append(name)
  results['accuracy_train'].append(train_score)
  results['accuracy_test'].append(test_score)
```


```python
results_df = pd.DataFrame(results)
format_dict = {'accuracy_train': '{:.2%}', 'accuracy_test': '{:.2%}'}
results_df.style.format(format_dict)
display(results_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>accuracy_train</th>
      <th>accuracy_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K-Nearest Neighbors</td>
      <td>0.955556</td>
      <td>0.983333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression</td>
      <td>0.955556</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SVM Linear kernel</td>
      <td>0.922222</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVM</td>
      <td>0.977778</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Decission Tree</td>
      <td>0.988889</td>
      <td>0.966667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest</td>
      <td>0.988889</td>
      <td>0.983333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AdaBoost</td>
      <td>0.955556</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GBM</td>
      <td>0.988889</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>XGBoost</td>
      <td>0.955556</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
boston_data = datasets.load_boston()
boston_df = pd.DataFrame(boston_data['data'])
boston_df.columns = boston_data['feature_names']
boston_df['target'] = boston_data['target']
```


```python
boston_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
from matplotlib import pyplot
```


```python
ggplot(boston_df, aes(x='LSTAT', y='target')) + geom_point()
```


    
<img src="/images/MachineLearning_intro_files/MachineLearning_intro_13_0.png">
    





    <ggplot: (8749825082681)>




```python
X_train, X_test, y_train, y_test = train_test_split(boston_df.drop('target', 1), boston_df['target'], test_size=0.4, random_state=42)
```


```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

```


```python
from matplotlib import pyplot

regressors = {'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
              'Linear Regression': LinearRegression(),
              'SVR Linear': SVR(kernel='linear'),
              'SVR RBF': SVR(kernel='rbf'),
              'Decission Tree': DecisionTreeRegressor(max_depth=5),
              'Random Forest': RandomForestRegressor(max_depth=5, n_estimators=20),
              'AdaBoost': AdaBoostRegressor(),
              'GBM': GradientBoostingRegressor(n_estimators=20),
              'XGBoost': XGBRegressor(n_estimators=100)}

results = {'model':[], 'accuracy_train':[], 'accuracy_test':[]}

for name, regressor in regressors.items():
  print(name)
  regressor.fit(X_train, y_train)
  train_score = regressor.score(X_train, y_train)
  test_score = regressor.score(X_test, y_test)
  yhat = regressor.predict(X_test)
  scatter = pyplot.scatter(y_test, yhat)
  display(scatter)
  results['model'].append(name)
  results['accuracy_train'].append(train_score)
  results['accuracy_test'].append(test_score)
```

    K-Nearest Neighbors



    <matplotlib.collections.PathCollection at 0x7f53a370e550>


    Linear Regression



    <matplotlib.collections.PathCollection at 0x7f53a370e690>


    SVR Linear



    <matplotlib.collections.PathCollection at 0x7f53a36f1450>


    SVR RBF



    <matplotlib.collections.PathCollection at 0x7f53a36f1050>


    Decission Tree



    <matplotlib.collections.PathCollection at 0x7f53a36f1890>


    Random Forest



    <matplotlib.collections.PathCollection at 0x7f53a3750690>


    AdaBoost



    <matplotlib.collections.PathCollection at 0x7f53a370ff50>


    GBM



    <matplotlib.collections.PathCollection at 0x7f53a36ac850>


    XGBoost
    [09:19:02] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.



    <matplotlib.collections.PathCollection at 0x7f53a36ac490>



    
<img src="/images/MachineLearning_intro_files/MachineLearning_intro_16_18.png">
    



```python
results_regression_df = pd.DataFrame(results)
format_dict = {'accuracy_train': '{:.2%}', 'accuracy_test': '{:.2%}'}
results_regression_df.style.format(format_dict)
display(results_regression_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>accuracy_train</th>
      <th>accuracy_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>K-Nearest Neighbors</td>
      <td>0.621195</td>
      <td>0.552343</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Linear Regression</td>
      <td>0.747200</td>
      <td>0.712514</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SVR Linear</td>
      <td>0.709390</td>
      <td>0.668472</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVR RBF</td>
      <td>0.178317</td>
      <td>0.240108</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Decission Tree</td>
      <td>0.936638</td>
      <td>0.751590</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Random Forest</td>
      <td>0.923696</td>
      <td>0.844382</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AdaBoost</td>
      <td>0.911834</td>
      <td>0.792974</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GBM</td>
      <td>0.912338</td>
      <td>0.827823</td>
    </tr>
    <tr>
      <th>8</th>
      <td>XGBoost</td>
      <td>0.976873</td>
      <td>0.877977</td>
    </tr>
  </tbody>
</table>
</div>



```python

```
