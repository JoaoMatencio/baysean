import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def predict(trace, X_test):
    beta_mean = trace.posterior["beta"].values.mean(axis=(0, 1))
    intercept_mean = trace.posterior["intercept"].values.mean(axis=(0, 1))
    p_test = pm.math.sigmoid(pm.math.dot(X_test, beta_mean) + intercept_mean)
    prob_test_predictions = p_test.eval()
    binary_test_predictions = (prob_test_predictions > 0.5).astype(int)
    return binary_test_predictions


def evaluate_predictions_with_negatives(y_true, y_pred):
    metrics = {
        "Precisão Positivo": precision_score(y_true, y_pred, pos_label=1),
        "Recall Positivo": recall_score(y_true, y_pred, pos_label=1),
        "F1-Score Positivo": f1_score(y_true, y_pred, pos_label=1),
        "Precisão Negativo": precision_score(y_true, y_pred, pos_label=0),
        "Recall Negativo": recall_score(y_true, y_pred, pos_label=0),
        "F1-Score Negativo": f1_score(y_true, y_pred, pos_label=0),
        "Acurácia": accuracy_score(y_true, y_pred),
    }
    return metrics

df = pd.read_csv("german_credit_data_preprocessed.csv")
df_train = df.sample(frac=0.8, random_state=19)
df_test = df.drop(df_train.index)

X_train = df_train.drop(columns=["pagamento_realizado"])
y_train = df_train["pagamento_realizado"]
X_test = df_test.drop(columns=["pagamento_realizado"])
y_test = df_test["pagamento_realizado"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["idade", "quantidade_credito", "duracao"]),
        ("cat", OneHotEncoder(), ["sexo", "trabalho", "moradia", "conta_poupanca", "conta_corrente", "objetivo"]),
    ]
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)
y = y_train.values

with pm.Model() as nuts_model:
    beta = pm.Normal("beta", mu=0, sigma=10, shape=X_train_preprocessed.shape[1])
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    linear_combination = pm.math.dot(X_train_preprocessed, beta) + intercept
    p = pm.Deterministic("p", pm.math.sigmoid(linear_combination))
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)
    nuts_trace = pm.sample(800, tune=800, cores=1, progressbar=True)

with pm.Model() as mh_model:
    beta = pm.Normal("beta", mu=0, sigma=10, shape=X_train_preprocessed.shape[1])
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    linear_combination = pm.math.dot(X_train_preprocessed, beta) + intercept
    p = pm.Deterministic("p", pm.math.sigmoid(linear_combination))
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)
    mh_trace = pm.sample(30000, tune=30000, step=pm.Metropolis(), cores=1, progressbar=True)

with pm.Model() as vi_model:
    beta = pm.Normal("beta", mu=0, sigma=10, shape=X_train_preprocessed.shape[1])
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    linear_combination = pm.math.dot(X_train_preprocessed, beta) + intercept
    p = pm.Deterministic("p", pm.math.sigmoid(linear_combination))
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)
    vi_approx = pm.fit(n=20000, method="advi")
    vi_trace = vi_approx.sample(1000)

nuts_predictions = predict(nuts_trace, X_test_preprocessed)
mh_predictions = predict(mh_trace, X_test_preprocessed)
vi_predictions = predict(vi_trace, X_test_preprocessed)

nuts_results = evaluate_predictions_with_negatives(y_test, nuts_predictions)
mh_results = evaluate_predictions_with_negatives(y_test, mh_predictions)
vi_results = evaluate_predictions_with_negatives(y_test, vi_predictions)

results_df = pd.DataFrame({
    "Métrica": nuts_results.keys(),
    "NUTS": nuts_results.values(),
    "Metropolis-Hastings": mh_results.values(),
    "Variational Inference": vi_results.values(),
})

import arviz as az

az.plot_trace(nuts_trace)
az.plot_trace(mh_trace)
az.plot_trace(vi_trace)

summary = az.summary(nuts_trace, round_to=2)
print(summary)

summary = az.summary(mh_trace, round_to=2)
print(summary)


summary = az.summary(vi_trace, round_to=2)
print(summary)