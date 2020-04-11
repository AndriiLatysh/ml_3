import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics as sm
import sklearn.model_selection as ms


def plot_model(model, qualifies_double_grade_df):
    plt.xlabel("Technical grade")
    plt.ylabel("English grade")

    qualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 0]

    max_grade = 101
    probability_level = np.empty([max_grade, max_grade])
    for technical_grade in range(max_grade):
        for english_grade in range(max_grade):
            prediction_point = [[technical_grade, english_grade]]
            probability_level[technical_grade, english_grade] = model.predict_proba(prediction_point)[:, 1]

    plt.contourf(probability_level, cmap="rainbow")

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")


qualifies_double_grade_df = pd.read_csv("data/double_grade.csv")

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

number_of_folds = 4

cv_qualification_model = lm.LogisticRegression()
cv_model_quality = ms.cross_val_score(cv_qualification_model, X, y, cv=number_of_folds, scoring="accuracy")
print(cv_model_quality)

prediction_model_quality = ms.cross_val_predict(cv_qualification_model, X, y, cv=number_of_folds)
cv_confusion_matrix = sm.confusion_matrix(y, prediction_model_quality)
print(cv_confusion_matrix)

qualification_model = lm.LogisticRegression()
qualification_model.fit(X, y)

modeled_qualification_probabilities = qualification_model.predict_proba(X)[:, 1]
qualifies_double_grade_df["modeled probability"] = modeled_qualification_probabilities

pd.set_option("display.max_rows", None)
print(qualifies_double_grade_df.sort_values(by="modeled probability"))

print(qualification_model.coef_)
print(qualification_model.intercept_)

plot_model(qualification_model, qualifies_double_grade_df)

plt.show()
