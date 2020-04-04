import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sk_lm
import mlinsights.mlmodel as mi_models


plt.figure(figsize=(20, 8))

call_center_df = pd.read_csv("data/call_center.csv", parse_dates=["timestamp"])

# plt.plot(call_center_df[["timestamp"]], call_center_df[["calls"]])

# call_center_df.at[17, "calls"] = 500
# call_center_df.at[18, "calls"] = 500
# call_center_df.at[19, "calls"] = 500

# X = np.array([t.value for t in call_center_df["timestamp"]]).reshape(-1, 1)
X = np.array(call_center_df.index).reshape(-1, 1)
y = np.array(call_center_df[["calls"]])

plt.plot(X, y, color="b")

border_values = np.array([X[0][0], X[-1][0]]).reshape(-1, 1)

print("OLS:")

ols_model = sk_lm.LinearRegression()
ols_model.fit(X, y)

ols_trend = ols_model.predict(border_values)

plt.plot(border_values, ols_trend, color="r")

print("Slope: {}".format(ols_model.coef_[0][0]))
print("Overall change: {}".format(ols_trend[1][0] - ols_trend[0][0]))

print("LAD:")

y = np.array(call_center_df["calls"])

lad_model = mi_models.QuantileLinearRegression()
lad_model.fit(X, y)

lad_trend = lad_model.predict(border_values)

plt.plot(border_values, lad_trend, color="g")

print("Slope: {}".format(lad_model.coef_[0]))
print("Overall change: {}".format(lad_trend[1] - lad_trend[0]))

plt.show()
