import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import sklearn.naive_bayes as sk_naive_bayes


def plot_model(model, qualifies_double_grade_df):
    plt.xlabel("Technical grade")
    plt.ylabel("English grade")

    qualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 0]

    max_grade = 101
    prediction_points = []

    for english_grade in range(max_grade):
        for technical_grade in range(max_grade):
            prediction_points.append([technical_grade, english_grade])

    probability_levels = model.predict_proba(prediction_points)[:, 1]
    probability_matrix = probability_levels.reshape(max_grade, max_grade)

    plt.contourf(probability_matrix, cmap="rainbow")  # cmap="RdYlBu"/"binary"

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")


qualifies_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

sns.pairplot(qualifies_double_grade_df, hue="qualifies")

k_folds = ms.StratifiedKFold(n_splits=4, shuffle=True)

naive_bayes_model = sk_naive_bayes.GaussianNB()
cv_predictions = ms.cross_val_predict(naive_bayes_model, X, y, cv=k_folds)

confusion_matrix = metrics.confusion_matrix(y, cv_predictions)
print(confusion_matrix)

naive_bayes_model.fit(X, y)

plt.figure()
plot_model(naive_bayes_model, qualifies_double_grade_df)

plt.show()
