import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics as sm


qualifies_single_grade_df = pd.read_csv("data/single_grade.csv")
qualifies_single_grade_df.sort_values(by=["grade", "qualifies"], inplace=True)

qualified_candidates = qualifies_single_grade_df[qualifies_single_grade_df["qualifies"] == 1]
unqualified_candidates = qualifies_single_grade_df[qualifies_single_grade_df["qualifies"] == 0]

plt.scatter(qualified_candidates["grade"], qualified_candidates["qualifies"], color="g")
plt.scatter(unqualified_candidates["grade"], unqualified_candidates["qualifies"], color="r")

X = qualifies_single_grade_df[["grade"]]
y = qualifies_single_grade_df["qualifies"]

qualification_model = lm.LogisticRegression()
qualification_model.fit(X, y)

modeled_qualification = qualification_model.predict(X)
modeled_qualification_probability = qualification_model.predict_proba(X)[:, 1]

qualifies_single_grade_df["modeled probability"] = modeled_qualification_probability
print(qualifies_single_grade_df)

plt.plot(X, modeled_qualification, color="b")
plt.plot(X, modeled_qualification_probability, color="c")

confusion_matrix = sm.confusion_matrix(y, modeled_qualification)
print(confusion_matrix)

plt.show()
