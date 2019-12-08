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








































        
    

















































































































