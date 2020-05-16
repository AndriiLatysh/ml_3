import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preprocessing
import sklearn.decomposition as sk_decomposition
import sklearn.linear_model as sk_linear
import sklearn.model_selection as sk_model_selection


iris_df = pd.read_csv("data/iris.csv").sample(frac=1)
column_names = iris_df.columns.tolist()

X = iris_df[column_names[:-1]]
y = iris_df[column_names[-1]]

standart_scaler = sk_preprocessing.StandardScaler()
X = standart_scaler.fit_transform(X)

label_encoder = sk_preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)

cv_iris_log_model = sk_linear.LogisticRegression()
cv_iris_model_quality = sk_model_selection.cross_val_score(cv_iris_log_model, X, y, cv=4, scoring="accuracy")

print("Model quality:")
print(cv_iris_model_quality)
print(np.mean(cv_iris_model_quality))

pca_2_components = sk_decomposition.PCA(n_components=2)
principal_components = pca_2_components.fit_transform(X)

iris_2_components_df = pd.DataFrame(data=principal_components, columns=["pc 1", "pc 2"])

iris_2_components_df["class"] = y

plt.scatter(x=iris_2_components_df["pc 1"], y=iris_2_components_df["pc 2"], c=iris_2_components_df["class"].values,
            cmap="prism")

cv_iris_2_log_model = sk_linear.LogisticRegression()
cv_iris_2_model_quality = sk_model_selection.cross_val_score(cv_iris_2_log_model, iris_2_components_df[["pc 1", "pc 2"]],
                                                             iris_2_components_df["class"], cv=4, scoring="accuracy")

print("2 components model quality:")
print(cv_iris_2_model_quality)
print(np.mean(cv_iris_2_model_quality))

pca_all_components = sk_decomposition.PCA()
pca_all_components.fit_transform(X)

print("Explained variance ratio:")
print(pca_all_components.explained_variance_)
print(pca_all_components.explained_variance_ratio_)

plt.figure()
components = list(range(1, pca_all_components.n_components_ + 1))
plt.plot(components, np.cumsum(pca_all_components.explained_variance_ratio_), marker="o")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.ylim(0, 1.1)

plt.show()
