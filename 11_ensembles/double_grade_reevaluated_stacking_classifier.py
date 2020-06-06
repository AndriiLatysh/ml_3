import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_trees
import sklearn.ensemble as sk_ensemble
import sklearn.neural_network as sk_neural_networks
import sklearn.svm as sk_svm
import sklearn.preprocessing as sk_preprocessing
import sklearn.model_selection as sk_model_selection


def plot_model(model, qualifies_double_grade_df, scaler, title, subplot):
    print("Starting plotting {}".format(title))

    subplot.set_title(title)

    qualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade_df[qualifies_double_grade_df["qualifies"] == 0]

    max_grade = 101
    prediction_points = []

    for english_grade in range(max_grade):
        for technical_grade in range(max_grade):
            prediction_points.append([technical_grade, english_grade])

    prediction_points = scaler.transform(prediction_points)
    probability_levels = model.predict_proba(prediction_points)[:, 1]
    probability_matrix = probability_levels.reshape(max_grade, max_grade)

    subplot.contourf(probability_matrix, cmap="rainbow")  # cmap="RdYlBu"/"binary"

    subplot.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    subplot.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")

    print("Finished plotting {}".format(title))


fig, subplots = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

qualifies_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualifies_double_grade_df[["technical_grade", "english_grade"]]
y = qualifies_double_grade_df["qualifies"]

X_scaler = sk_preprocessing.StandardScaler()
X = X_scaler.fit_transform(X)

k_folds = sk_model_selection.StratifiedKFold(n_splits=4, shuffle=True)

ann_model = sk_neural_networks.MLPClassifier(hidden_layer_sizes=(10, 10), activation="tanh", max_iter=100000)
ann_result = sk_model_selection.cross_val_score(ann_model, X, y, cv=k_folds)
print("Neural Network accuracy: {:.2f} %".format(ann_result.mean() * 100))

svm_model = sk_svm.SVC(probability=True)
svm_result = sk_model_selection.cross_val_score(svm_model, X, y, cv=k_folds)
print("Support Vector Machine accuracy: {:.2f} %".format(svm_result.mean() * 100))

rfc_model = sk_ensemble.RandomForestClassifier(n_jobs=-1)
rfc_result = sk_model_selection.cross_val_score(rfc_model, X, y, cv=k_folds)
print("Random Forest accuracy: {:.2f} %".format(rfc_result.mean() * 100))

models = []
models.append(("ANN", ann_model))
models.append(("SVM", svm_model))
models.append(("RFC", rfc_model))

stacking_model = sk_ensemble.StackingClassifier(models, cv=k_folds)
stacking_results = sk_model_selection.cross_val_score(stacking_model, X, y, cv=k_folds)

print()
print("Stacking model accuracy: {:.2f} %".format(stacking_results.mean() * 100))
print()

ann_model.fit(X, y)
svm_model.fit(X, y)
rfc_model.fit(X, y)
stacking_model.fit(X, y)

plot_model(ann_model, qualifies_double_grade_df, X_scaler, "Neural Network", subplots[0][0])
plot_model(svm_model, qualifies_double_grade_df, X_scaler, "Support Vector Machine", subplots[0][1])
plot_model(rfc_model, qualifies_double_grade_df, X_scaler, "Random Forest", subplots[0][2])
plot_model(stacking_model, qualifies_double_grade_df, X_scaler, "Stacking model", subplots[1][1])

plt.show()
