# Predictive-Analytics-in-Python
- Build ML model with meaningful variables. Use model for predictions.
- **Predictive analytics** is an process that aims at predicting an event using historical data. This data is gathered in the analytical basetable.

### Analytical Basetable structure
- An **analytical base table** is typically stored in a pandas dataframe. Three important variables in the analytical basetable are : **`population`**, **`candidate predictors`** and the **`target`**
- **Population** is the group of people or object we want to make the predicton for **(rows of data)**
- **Candidate predictors** is the information that can be used to predict the event **(features)**
- **Target** is the event to **`predict`**, 

```python
import pandas as pd
basetable = pd.DataFrame("import_basetable.csv")
population_size = len(basetable)
targets = sum(basetable["target"])
```

### Logistic Regression

```python
from sklearn import linear_model
logreg = linear_model.LogisticRegression()
X = basetable[['age']]
y = basetable[['target']]
logreg.fit(X, y)
print(logreg.coef_)
print(logreg.intercept_)

```

### Multivariate logistic regression
- Univariate : ax+b
- Multivariate : a1x1 + a2x2 + a3x3 +....+ anxn + b

### Making predictions

```python
new_data = current_data[["gender_F", "ag", "time_since_last_gift"]]
predictions = logreg.predict(new_data)
```

### Variable selection

#### Drawbacks of models with many variables
- Overfitting
- Hard to maintain or implement
- Hard to interpret, multi-collinearity : correalted variables make interpretation harder

- The goal of variable selection is to select a set of variables that has optimal performance

### Model evaluation : AUC
- A metric often used to quantify the model performance is AUC value. It is a value between 0 & 1, "1" being the perfect model.

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(true_target, prob_target)
```

- **`the model with 5 variables has the same AUC as the model using only 2 variables. Adding more variables doesn't always increase the AUC`**

- **`AUC score can be used to determine whether increasing or decreasing the model variables increases the performance or not`**


### Forward stepwise variable selection : Intutive way of variable selection
- The forward stepwise variable selection procedure:
**First** it selects among all the candidate predictors the variable that has the best AUC when used in the model. **Next**, it selects another candidate predictor that has the best AUC incombination with the first selected variable. This continues until all variables are added or until predefined number of variables is added.
- Find best variable **v1**
- Find best variable **v2** in combination with **v1**
- Find best variable **v3** in combination with **v1,v2**
- **Until all variables are added or until predefined number of variables is added**

### Implementation of the forward stepwise variable selection
- Function **auc** that calculates AUC given a certain set of variables
- Function **best_next** that returns next best variable in combination with current variables
- Loop until desired number of variables


### Implementation of AUC function

```python
from sklearn import linear_model
from sklearn.metrics import roc_auc_score

def auc(variables, target, basetable):
    X = basetable[variables]
    y = basetable[target]
    
    logreg = linear_model.LogisticRegression()
    logreg.fit(X,y)
    
    predictions = logreg.predict_proba(X)[:,1]
    auc = roc_auc_score(y, predictions)
    return auc
```

- calling **auc**

```python
auc = auc(["age","gender_F"], ["target"], basetable)
print(round(auc, 2))
```

### Calculating the next best vriable

```python
def next_best(current_variable, candidate_variable, target, basetable):
    """function looks throughout candidate variables and keeps track of which  variable is best and the auc associated with the best variable"""
    best_auc = -1
    best_variable = None
    
    # for each variable in the candidate variable set calculate the AUC
    # current_variable : variables which are already in the model
    # extend it with the variable with which we need to evaluate
    for v in candidate_variables:
        auc_v = auc(current_variables + [v], target, basetable)
        
    # if this AUC is better then the best AUC, change the best AUC and best variable
    if auc_v >= best_auc:
        best_auc = auc_v
        best_variable = v
    return best_variable
```

- If we want to know which variable among `min_gift, max_gift, mean_gift` should be added next given that `age and gender_F` are already in the model, we can use **`next_best`** function as follows;

```python
current_variables = ["age", "gender_F"]
candidate_variables = ["min_gift", "max_gift", "mean_gift"]
next_variable = next_best(current_variables, candidate_variables, basetable)
print(next_varible)
```

- To complete the forward stepwise variable selection procedure, we keep track of the candidate variables and current variables added to the model so far

```python
candidate_variables = ["mean_gift", "min_gift", "max_gift","age","gender_F", "country_USA", "income_low"]
current_variables = []
target = ["target"]
```

- We can define the max number of variables that can be added. In each iteration, the next_best variable is calculated using the next_best function. The current variable list is updated by already chosen variable and the chosen variable is removed from the candidate variable list.

```python
max_number_variables = 5
number_iterations = min(max_number_variables, len(candidate_variables))
for i in range(0, number_iterations):
    next_var = next_best(current_variables, candidate_variables, target, basetable)
    current_variables = current_variables + [next_variable]
    candidate_variables.remove(next_variable)
    
print(current_variables)
```

### Deciding on the number of variables
- Forward Stepwise variable selection returns the order in which the variables increase the accuracy, but we need to decide on how many variables to use.

```python
auc_values = []
variables_evaluate = []

for v in variables_forward:
    variables_evaluate.append(v)
    auc_value = auc(variables_evaluate, ["target"], basetable)
    auc_values.append(auc_value)
```

- Inorder to do so, we can have a look at the **AUC values**. The order of the variables is given in the list `variables_forward`. For each variable in variable forward calculate the AUC values.

<p align="center">
  <img src="data/AUC.JPG" width="350" title="AUC">
</p>

- If we plot the AUC values we obtain a curve that typically keeps increasing
. However, if we use new data to evaluate subsequent models it doesn't increase, instead it decreases after a while. This phenomenon is called overfitting.
- By adding more variables the accuracy on the data on which model is built increases, but the true performance of the model decreases because the complex model doesnt generalize to other data.

#### Detecting over-fitting
- There exits smart techniques to detect and prevent overfitting. Performance on the test dataset is representative of the true performance of the model.
- One way of partioning data is randomly dividing the data into two parts, however when the data is imbalanced it is important to make sure that the target variable is in same proportion in train and test. It can be done by using **`stratify`** on the target while splitting the data.

```python
from sklearn.model_selection import train_test_split

X = basetable.drop("target", 1)
y = basetable["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify = Y)

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
```

#### Deciding the cut-off

<p align="center">
  <img src="data/cutoff.JPG" width="350" title="cutoff">
</p>

- We can now plot AUC curves of the subsequent models on both the train and test data.We can see that the train AUC keeps increasing while the test AUC stabalizes and then decreases.
- When deciding on how many variables to keep in the model, once we take into account that **test AUC is as high as possible** and the **model should have least variables possible**.
- In this case it's clear that the cut-off indicated by the dashed line is the best option. All models having more variables has lower test accuracy.

## Explaining model performance to business

### The cumulative gains curve
- Once the model is ready we need to show it to the business. VIsualization of model performance that business people can understand.
- Until now we evaluated models using the AUC. Though it is very useful for data scientist it is less appropriate if we want to discuss the model performance with business stakeholders.
- Indeed, AUC is a bit complex evaluation measure that is not much intutive.Moreover, its a single number which doesn't catch all the information about the model.
- For better visualization we can use evaluation curve like the cumulative gains curve. This type of curves are easy to explain and guide us to better business decisions.

#### **Cumulative gains curve** is constructed as follows :

<p align="center">
  <img src="data/cumulative_gains.JPG" width="350" title="cumulative_gains">
</p>

- First, we order all the observations according to the output of the model. One the LHS are the observations with the highest probabilty to be target according to the model and on the RHS are the observations with lowest probabilty to be target.
- On the horizontal axis of cumulative gains curve, it is indicated which percentage of the observations is considered. For instance, 30% of the observations with the highest probabilty to be target is considered.
- On the vertical axis, the curve indicates which percentage of all targets is included in this group. For instance, if the cumulative gain is 70% at 30%, it means we are taking the top 30% observations with highest probabilty to be target, this group contains already 70% of all targets. 

<p align="center">
  <img src="data/cum_gain.JPG" width="350" title="cumulative_gains">
</p>

- The cumulative gains curve is the great tool to compare models. **The more the line is situated to the upper left corner, the better the model**. It is often the case that two models produce curves that cross each other. In that case, it is not straightforward to decide which model is best. In this case, for instance, we can say that model 2 is better to distinguish the top 10% observations from the rest, while model 1 is better to distinguish the top 70% of the observations from the rest.

#### Cumulative gains in python
- Constructing cumulative gains curves in Python is easy with the **scikitplot** module.

```python
import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_cumulative_gain(true_values, predictions)
plt.show()
```








































































































        
    

















































































































