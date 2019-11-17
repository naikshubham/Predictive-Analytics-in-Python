# Predictive-Analytics-in-Python
- Build ML model with meaningful variables. Use model for predictions.
- **Predictive analytics** is an process that aims at predicting an event using historical data. This data is gathered in the analytical basetable.

### Analytical Basetable structure
- An **analytical base table** is typically stored in a pandas dataframe. Three important variables in the analytical basetable are : **`population`**, **`candidate predictors`** and the **`target`**
- **Population** is the group of people or object we want to make the predicton for **(rows of data)**
- **Candidate predictors** is the information that can be used to predict the event **(features)**
- **Tagret** is the event to **`predict`**, 

```python
import pandas as pd
basetable = pd.DataFrame("import_basetable.csv")
population_size = len(basetable)
targets = sum(basetable["target"])
```

