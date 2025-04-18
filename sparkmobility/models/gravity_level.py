import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
# Informal 0
data = {
    "Origin": ["A", "A", "C", "E", "E"],
    "Destination": ["B", "B", "D", "F", "F"],
    "Flow_Type": [0, 0, 0, 1, 1], 
    "Flow": [100, 200, 150, 80, 120],
    "Origin_Mass": [500000, 500000, 300000, 200000, 200000],
    "Destination_Mass": [1000000, 1000000, 800000, 600000, 600000],
    "Distance": [50, 50, 100, 30, 30],
}

df = pd.DataFrame(data)

df["log_Flow"] = np.log(df["Flow"])
df["log_Origin_Mass"] = np.log(df["Origin_Mass"])
df["log_Destination_Mass"] = np.log(df["Destination_Mass"])
df["log_Distance"] = np.log(df["Distance"])

formula = "log_Flow ~ log_Origin_Mass + log_Destination_Mass + log_Distance + (1 + log_Origin_Mass + log_Destination_Mass + log_Distance | Flow_Type)"

model = smf.mixedlm(formula, df, groups=df["Flow_Type"])
result = model.fit()

print(result.summary())