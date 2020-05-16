import pandas as pd
import numpy as np
import sklearn.neighbors as sk_neighbours
import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt


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

    plt.contourf(probability_level, cmap="rainbow")  # cmap="RdYlBu"/"binary"

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")


qualifies_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

for k in range(1, 10, 2):
    print(f"{k} neighbours:")

    double_grade_knn_model = sk_neighbours.KNeighborsClassifier(n_neighbors=k)
    cv_double_grade_knn_prediction = sk_model_selection.cross_val_predict(double_grade_knn_model, X, y, cv=4)

    cv_confusion_matrix = sk_metrics.confusion_matrix(y, cv_double_grade_knn_prediction)
    print(cv_confusion_matrix)

double_grade_knn_model = sk_neighbours.KNeighborsClassifier(n_neighbors=3)
double_grade_knn_model.fit(X, y)

plot_model(double_grade_knn_model, qualifies_double_grade_df)

plt.show()
