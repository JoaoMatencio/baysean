import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import shap
import pandas as pd
import numpy as np

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

numeric_features = ["idade", "quantidade_credito", "duracao"]
categorical_features = ["sexo", "trabalho", "moradia", "conta_poupanca", "conta_corrente", "objetivo"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough"
)

df_train = df.sample(frac=0.8, random_state=19)
df_test = df.drop(df_train.index)

X_train = df_train.drop(columns=["pagamento_realizado"])
y_train = df_train["pagamento_realizado"]
X_test = df_test.drop(columns=["pagamento_realizado"])
y_test = df_test["pagamento_realizado"]

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

decision_tree = DecisionTreeClassifier(random_state=19, max_depth=3)
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_model = lgb.train(
    {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": 19,
    },
    lgb_train,
    valid_sets=[lgb_train],
    num_boost_round=100
)
y_pred_lgb = (lgb_model.predict(X_test) > 0.5).astype(int)

tree_metrics = evaluate_predictions_with_negatives(y_test, y_pred_tree)
lgb_metrics = evaluate_predictions_with_negatives(y_test, y_pred_lgb)

metrics_df = pd.DataFrame({
    "Árvore de Decisão": tree_metrics,
    "LightGBM": lgb_metrics
})


explainer = shap.TreeExplainer(lgb_model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="dot")

shap.summary_plot(shap_values, X_test, plot_type="dot")

if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())

shap_values = explainer(X_test)

shap.plots.waterfall(shap_values[10])

