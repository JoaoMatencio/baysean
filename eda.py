import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_categorical(column_data, column_name):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Criar eixo secundário
    ax2 = ax1.twinx()

    # Plotando porcentagem no eixo à esquerda
    percentage = (column_data.value_counts(normalize=True) * 100).sort_values(ascending=False)
    ax1.bar(percentage.index, percentage.values, color='skyblue', alpha=0.7, label='Porcentagem')
    ax1.set_ylabel('Porcentagem (%)')
    ax1.set_xlabel(column_name)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper left')

    # Plotando volumetria no eixo à direita
    volume = column_data.value_counts().sort_values(ascending=False)
    ax2.plot(volume.index, volume.values, color='blue', marker='o', label='Frequência')
    ax2.set_ylabel('Frequência')
    ax2.legend(loc='upper right')

    # Título do gráfico
    plt.title(f'Distribuição de {column_name}')
    plt.tight_layout()
    plt.show()

def plot_numerical(column_data, column_name):
    plt.figure(figsize=(8, 6))
    sns.histplot(column_data, kde=True, bins=20, color='blue')
    plt.title(f'Distribuição de {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.show()

def plot_data_types(df):
    for column in df.columns:
        if pd.api.types.is_categorical_dtype(df[column]):
            print(f"Plotando variável categórica: {column}")
            plot_categorical(df[column], column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            print(f"Plotando variável numérica: {column}")
            plot_numerical(df[column], column)

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

plot_data_types(df)

df["pagamento_realizado"] = np.where(
    ((df["quantidade_credito"] < 5000) & (df["duracao"] < 12)) |
    ((df["idade"] < 30) & (df["sexo"] == "masculino") & (df["trabalho"] == 2)) |
    ((df["idade"] < 30) & (df["sexo"] == "feminino") & (df["trabalho"] == 1)) |
    ((df["idade"] > 40) & (df["sexo"] == "masculino") & (df["trabalho"] == 3)) |
    ((df["idade"] > 40) & (df["sexo"] == "feminino") & (df["trabalho"] == 2)) |
    ((df["quantidade_credito"] > 10000) & (df["duracao"] > 24)),
    1,  # Pagamento realizado
    0   # Pagamento não realizado
)

indices_to_remove = df[df["pagamento_realizado"].notna()].sample(n=100, random_state=42).index
df.loc[indices_to_remove, "pagamento_realizado"] = np.nan

df.to_csv("german_credit_data_preprocessed.csv", index=False)