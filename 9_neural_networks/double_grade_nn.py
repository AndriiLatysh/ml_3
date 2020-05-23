import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preprocessing
import sklearn.neural_network as sk_nn
import sklearn.model_selection as sk_ms
import sklearn.metrics as sk_metrics


def plot_model(model, qualifies_double_grade_df, input_scaler):
    plt.xlabel("Technical grade")
    plt.ylabel("English grade")

    qualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 0]

    max_grade = 101
    prediction_points = []

    for english_grade in range(max_grade):
        for technical_grade in range(max_grade):
            prediction_points.append([technical_grade, english_grade])
    prediction_points = input_scaler.transform(prediction_points)

    probability_levels = model.predict_proba(prediction_points)[:, 1]
    probability_matrix = probability_levels.reshape(max_grade, max_grade)

    plt.contourf(probability_matrix, cmap="rainbow")  # cmap="RdYlBu"/"binary"

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")


qualifies_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df[["qualifies"]]

min_max_scaler = sk_preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

one_hot_encoding = sk_preprocessing.OneHotEncoder(sparse=False)
y = one_hot_encoding.fit_transform(y)

X_train, X_test, y_train, y_test = sk_ms.train_test_split(X, y)

qualification_model = sk_nn.MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=1000000)
qualification_model.fit(X_train, y_train)

y_predicted = qualification_model.predict(X_test)

transformed_y_test = one_hot_encoding.inverse_transform(y_test)
transformed_y_predicted = one_hot_encoding.inverse_transform(y_predicted)

confusion_matrix = sk_metrics.confusion_matrix(transformed_y_test, transformed_y_predicted)
print(confusion_matrix)

plot_model(qualification_model, qualifies_double_grade_df, min_max_scaler)

plt.show()
