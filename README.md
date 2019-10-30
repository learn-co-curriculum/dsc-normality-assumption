
## Log Transformations

## Introduction

In this lesson, you will take a look at logarithmic transformations and when to apply them to features of a dataset. This will then become an effective technique you can use to improve the performance of linear regression models. Remember, linear regression models are meant to determine optimal coefficients in order to decompose an output variable as the linear combination of features. Transforming these initial features to have certain properties such as normality will improve the regression algorithms predictive performance.

## Objectives

You will be able to:
* Identify if it is necessary to perform log transformations on a set of features
* Perform log transformations on different features of a dataset


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Linear Regression Assumptions

Remember that linear regression operates under various assumptions including that the dependent variable can be decomposed into a linear combination of the independent features. Additionally, data should be homoscedastic and the residuals should follow a normal distribution.

One thing we briefly touched upon previously is the **distributions of the predictors**. In previous labs, you have looked at these distributions to have an understanding of what the distributions look like. In fact, you'll often find that having the data more normally distributed will benefit your model and model performance in general. So while normality of the predictors is not a mandatory assumption, having (approximately) normal features may be helpful for your model!

## A Model Using the Raw Features

To prove the point, let's look at a model using raw inputs that are not approximately normal. Afterwards, you'll take a look at how to identify when you can **transform your inputs** (log transformations) and validate the improvement that they provide for the model.


```python
data = pd.read_csv('auto-mpg.csv')
data.head()
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
      <th>Unnamed: 0</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from statsmodels.formula.api import ols
```


```python
outcome = 'mpg'
x_cols = ['displacement', 'horsepower', 'weight', 'acceleration']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=data).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.707</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.704</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   233.4</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 01 Oct 2019</td> <th>  Prob (F-statistic):</th> <td>9.63e-102</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:42:44</td>     <th>  Log-Likelihood:    </th> <td> -1120.6</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   2251.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   387</td>      <th>  BIC:               </th> <td>   2271.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>   45.2511</td> <td>    2.456</td> <td>   18.424</td> <td> 0.000</td> <td>   40.422</td> <td>   50.080</td>
</tr>
<tr>
  <th>displacement</th> <td>   -0.0060</td> <td>    0.007</td> <td>   -0.894</td> <td> 0.372</td> <td>   -0.019</td> <td>    0.007</td>
</tr>
<tr>
  <th>horsepower</th>   <td>   -0.0436</td> <td>    0.017</td> <td>   -2.631</td> <td> 0.009</td> <td>   -0.076</td> <td>   -0.011</td>
</tr>
<tr>
  <th>weight</th>       <td>   -0.0053</td> <td>    0.001</td> <td>   -6.512</td> <td> 0.000</td> <td>   -0.007</td> <td>   -0.004</td>
</tr>
<tr>
  <th>acceleration</th> <td>   -0.0231</td> <td>    0.126</td> <td>   -0.184</td> <td> 0.854</td> <td>   -0.270</td> <td>    0.224</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>38.359</td> <th>  Durbin-Watson:     </th> <td>   0.861</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  51.333</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.715</td> <th>  Prob(JB):          </th> <td>7.13e-12</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.049</td> <th>  Cond. No.          </th> <td>3.56e+04</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.56e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



## Checking Variable Distributions

You do have an initial model displayed above, but this can be improved. The first step you should take prior to simply fitting your model is to see how each of the variables are related to one another.


```python
pd.plotting.scatter_matrix(data[x_cols], figsize=(10,12));
```


![png](index_files/index_8_0.png)


## Logarithmic Functions

As you'll see below, one common option for transforming non-normal variable distributions is to try applying a logarithmic function and observe its impact of the distribution. As a helpful math review, let's take a look at a logarithmic curve. (Also remember that you can't take the logarithm of zero nor a negative number.)


```python
x = np.linspace(start=-100, stop=100, num=10**3)
y = np.log(x)
plt.plot(x, y);
```

    //anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: RuntimeWarning: invalid value encountered in log
      



![png](index_files/index_10_1.png)


## Transforming Non-Normal Features


```python
non_normal = ['displacement', 'horsepower', 'weight']
for feat in non_normal:
    data[feat] = data[feat].map(lambda x: np.log(x))
pd.plotting.scatter_matrix(data[x_cols], figsize=(10,12));
```


![png](index_files/index_12_0.png)


## A Model After Transforming Non-Normal Features


```python
outcome = 'mpg'
x_cols = ['displacement', 'horsepower', 'weight', 'acceleration']
predictors = '+'.join(x_cols)
formula = outcome + '~' + predictors
model = ols(formula=formula, data=data).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>mpg</td>       <th>  R-squared:         </th> <td>   0.748</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.745</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   286.5</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 01 Oct 2019</td> <th>  Prob (F-statistic):</th> <td>2.98e-114</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:42:46</td>     <th>  Log-Likelihood:    </th> <td> -1091.4</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   392</td>      <th>  AIC:               </th> <td>   2193.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   387</td>      <th>  BIC:               </th> <td>   2213.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>  154.5685</td> <td>   12.031</td> <td>   12.847</td> <td> 0.000</td> <td>  130.913</td> <td>  178.223</td>
</tr>
<tr>
  <th>displacement</th> <td>   -3.2705</td> <td>    1.219</td> <td>   -2.684</td> <td> 0.008</td> <td>   -5.667</td> <td>   -0.874</td>
</tr>
<tr>
  <th>horsepower</th>   <td>  -11.0811</td> <td>    1.911</td> <td>   -5.800</td> <td> 0.000</td> <td>  -14.837</td> <td>   -7.325</td>
</tr>
<tr>
  <th>weight</th>       <td>   -7.2456</td> <td>    2.753</td> <td>   -2.632</td> <td> 0.009</td> <td>  -12.658</td> <td>   -1.834</td>
</tr>
<tr>
  <th>acceleration</th> <td>   -0.3760</td> <td>    0.131</td> <td>   -2.876</td> <td> 0.004</td> <td>   -0.633</td> <td>   -0.119</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>40.779</td> <th>  Durbin-Watson:     </th> <td>   0.972</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  64.330</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.674</td> <th>  Prob(JB):          </th> <td>1.07e-14</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.456</td> <th>  Cond. No.          </th> <td>1.17e+03</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.17e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



## Observations

While not dramatic, you can observe that simply by transforming non-normally distributed features using log transformations, we have increased our $R^2$ value of the model from 0.707 to 0.748. 

## Summary

In this lesson, you got a quick review of logarithmic functions, and saw how they can be used to transform non-normal distributions which can improve the performance of linear regression models.
