import pandas as pd
import sklearn.naive_bayes as sk_naive_bayes


def convert_to_numeric_values(df):
    converted_df = df.copy()
    converted_df = converted_df.replace({"history": {"bad": 0, "fair": 1, "excellent": 2},
                                         "income": {"low": 0, "high": 1},
                                         "term": {3: 0, 10: 1}})
    return converted_df


loan_df = pd.read_csv("data/loans.csv")

numeric_loan_df = convert_to_numeric_values(loan_df)
print(numeric_loan_df)

feature_names = loan_df.columns.values[:-1]
X = numeric_loan_df[feature_names]
y = numeric_loan_df["risk"]

naive_bayes_model = sk_naive_bayes.CategoricalNB()

naive_bayes_model.fit(X, y)

X_probabilities = naive_bayes_model.predict_proba(X)[:, 1]
X_probabilities_log = naive_bayes_model.predict_log_proba(X)[:, 1]

loan_df["probability"] = X_probabilities
loan_df["log_probability"] = X_probabilities_log

print(loan_df)
