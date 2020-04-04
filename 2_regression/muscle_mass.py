import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.linear_model as lm


muscle_mass_df = pd.read_csv("data/muscle_mass.csv")
muscle_mass_df.sort_values(by="training_time", inplace=True)

X = muscle_mass_df[["training_time"]]
y = muscle_mass_df[["muscle_mass"]]

plt.scatter(X, y)

polynomial_transformer = pp.PolynomialFeatures(degree=2)
X_transformed = polynomial_transformer.fit_transform(X)

print(X_transformed)

muscle_mass_model = lm.LinearRegression()
muscle_mass_model.fit(X_transformed, y)

print(muscle_mass_model.coef_)
print(muscle_mass_model.intercept_)

modeled_muscle_mass = muscle_mass_model.predict(X_transformed)

plt.plot(X, modeled_muscle_mass, color="r")

plt.show()
