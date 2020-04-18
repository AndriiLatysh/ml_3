import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import sklearn.metrics as sm
import sklearn.model_selection as ms
import double_grade_svm_utility


qualifies_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

double_grade_svm_utility.plot_values(qualifies_double_grade_df)

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

# svm_soft_non_linear_model = svm.SVC(kernel="rbf")
# svm_soft_non_linear_model.fit(X, y)
#
# double_grade_svm_utility.plot_model(svm_soft_non_linear_model)

parameter_grid = {"kernel": ["rbf"], "C": [10 ** p for p in range(-2, 6)], "gamma": [10 ** p for p in range(-6, 2)]}
# parameter_grid = {"kernel": ["rbf"], "C": [10 ** p for p in range(1, 7)], "gamma": [p * 1e-5 for p in range(1, 10)]}

grid_search = ms.GridSearchCV(svm.SVC(), param_grid=parameter_grid, cv=4)
grid_search.fit(X, y)

print(grid_search.best_params_)

modeled_qualification = grid_search.predict(X)
confusion_matrix = sm.confusion_matrix(y, modeled_qualification)

print(confusion_matrix)

double_grade_svm_utility.plot_model(grid_search.best_estimator_)

plt.show()
