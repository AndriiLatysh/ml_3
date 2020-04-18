import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import sklearn.metrics as sm
import sklearn.model_selection as ms
import double_grade_svm_utility


qualifies_double_grade_df = pd.read_csv("data/double_grade.csv")

double_grade_svm_utility.plot_values(qualifies_double_grade_df)

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

cv_svm_soft_linear_model = svm.SVC(kernel="linear")
cv_svm_soft_linear_predictions = ms.cross_val_predict(cv_svm_soft_linear_model, X, y, cv=4)

cv_confusion_matrix = sm.confusion_matrix(y, cv_svm_soft_linear_predictions)
print(cv_confusion_matrix)

svm_soft_linear_model = svm.SVC(kernel="linear")
svm_soft_linear_model.fit(X, y)

double_grade_svm_utility.plot_model(svm_soft_linear_model)

plt.show()
