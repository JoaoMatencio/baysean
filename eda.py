import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_categorical(data, categorical_column, hue_column="pagamento_realizado"):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=categorical_column, hue=hue_column, palette=["#1f77b4", "#ff7f0e"])
    plt.title(f"Distribuição de {categorical_column} por {hue_column}", fontsize=14)
    plt.xlabel(categorical_column, fontsize=12)
    plt.ylabel("Contagem", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title=hue_column)
    plt.tight_layout()
    plt.show()

def plot_numerical(data, numerical_column, hue_column="pagamento_realizado"):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=numerical_column, hue=hue_column, kde=False, palette=["#1f77b4", "#ff7f0e"], bins=30, multiple="stack")
    plt.title(f"Distribuição de {numerical_column} por {hue_column}", fontsize=14)
    plt.xlabel(numerical_column, fontsize=12)
    plt.ylabel("Densidade", fontsize=12)
    plt.xticks(fontsize=10)
    plt.legend(title=hue_column)
    plt.tight_layout()
    plt.show()


def plot_data_types(df):
    for column in df.columns:
        if pd.api.types.is_categorical_dtype(df[column]):
            print(f"Plotando variável categórica: {column}")
            plot_categorical(df, column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            print(f"Plotando variável numérica: {column}")
            plot_numerical(df, column)

df = pd.read_csv('german_credit_data.csv')

df["idade"] = df["Age"]
df["sexo"] = df["Sex"]
df["trabalho"] = df["Job"]
df["moradia"] = df["Housing"]
df["conta_poupanca"] = df["Saving accounts"]
df["conta_corrente"] = df["Checking account"]
df["quantidade_credito"] = df["Credit amount"]
df["duracao"] = df["Duration"]
df["objetivo"] = df["Purpose"]
df["trabalho"] = df["trabalho"].astype(object)

df = df.drop(columns=["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration", "Purpose", "Unnamed: 0"])

print(df.head())
df.fillna("Don't have", inplace=True)

translation_maps = {
    'objetivo': {
        'radio/TV': 'rádio/TV',
        'education': 'educação',
        'furniture/equipment': 'móveis/equipamentos',
        'car': 'carro',
        'business': 'negócios',
        'domestic appliances': 'eletrodomésticos',
        'repairs': 'reparos',
        'vacation/others': 'férias/outros'
    },
    'sexo': {
        'male': 'masculino',
        'female': 'feminino'
    },
    'moradia': {
        'own': 'própria',
        'free': 'gratuita',
        'rent': 'alugada'
    },
    'conta_poupanca': {
        "Don't have": 'não possui',
        'little': 'pouco',
        'moderate': 'moderado',
        'quite rich': 'bastante',
        'rich': 'rico'
    },
    'conta_corrente': {
        'little': 'pouco',
        'moderate': 'moderado',
        "Don't have": 'não possui',
        'rich': 'rico'
    }
}

def translate_column(df, column, translation_map):
    if column in df.columns:
        df[column] = df[column].map(translation_map)

for column, translation_map in translation_maps.items():
    translate_column(df, column, translation_map)


for coluna in df.columns:
    print(f"{coluna}: {df[coluna].unique()}")

for colunas_numericas in df.columns:
    if df[colunas_numericas].dtype == "int64":
        print(f"{colunas_numericas}: {df[colunas_numericas].describe()}")

for coluna in df.columns:
    if df[coluna].dtype == "object":
        df[coluna] = df[coluna].astype("category")


df["pagamento_realizado"] = np.where(
    ((df["quantidade_credito"] < 5000) & (df["duracao"] < 12)) |
    ((df["idade"] < 30) & (df["sexo"] == "masculino") & (df["trabalho"] == 2)) |
    ((df["idade"] < 30) & (df["sexo"] == "feminino") & (df["trabalho"] == 1)) |
    ((df["idade"] > 40) & (df["sexo"] == "masculino") & (df["trabalho"] == 3)) |
    ((df["idade"] > 40) & (df["sexo"] == "feminino") & (df["trabalho"] == 2)) |
    ((df["quantidade_credito"] > 10000) & (df["duracao"] > 24)),
    1,
    0
)

plot_data_types(df)

df.to_csv("german_credit_data_preprocessed.csv", index=False)