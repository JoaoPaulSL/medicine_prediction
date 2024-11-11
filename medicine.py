from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

# Carregar os dados dos arquivos CSV
sintomas_df = pd.read_csv('C:/Users/joaop/Desktop/python/medicines_ai/csv/doencas_sintomas.csv')
tratamentos_df = pd.read_csv('C:/Users/joaop/Desktop/python/medicines_ai/csv/doencas_tratamentos.csv')

# Verificar as colunas dos DataFrames para garantir que 'Doença' está presente
print("Colunas de sintomas_df:", sintomas_df.columns)
print("Colunas de tratamentos_df:", tratamentos_df.columns)

# Unindo os dois DataFrames com base na coluna 'Doença'
df = pd.merge(sintomas_df, tratamentos_df, on='Doença')  # Corrigir para 'Doença'

# Convertendo os sintomas em uma lista de características numéricas
vectorizer = CountVectorizer(stop_words='english')

# Convertendo sintomas em vetores numéricos
X = vectorizer.fit_transform(df['Sintomas']).toarray()  # Corrigir para 'Sintomas'
y = df['Doença']  # Corrigir para 'Doença'

# Codificando as doenças para rotular
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Se você deseja usar a função de perda 'categorical_crossentropy', use to_categorical
y_encoded_categorical = to_categorical(y_encoded)  # Caso escolha 'categorical_crossentropy'

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Redes Neurais
nn_model = Sequential()
nn_model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

# Se estiver usando 'sparse_categorical_crossentropy', a variável y_encoded deve ser usada diretamente
nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

nn_model.fit(X_train, y_train, epochs=10, batch_size=32)

# Função para prever a doença e tratamento com feedback adicional
def prever_doenca_tratamento(symptoms_text):
    # Transformar o texto do paciente em vetor numérico
    symptoms_vector = vectorizer.transform([symptoms_text]).toarray()

    # Previsão com KNN
    knn_prob = knn_model.predict_proba(symptoms_vector)
    knn_pred = knn_model.predict(symptoms_vector)
    knn_doenca = label_encoder.inverse_transform(knn_pred)[0]
    knn_confidence = knn_prob.max()  # Confiança do KNN

    # Previsão com Random Forest
    rf_prob = rf_model.predict_proba(symptoms_vector)
    rf_pred = rf_model.predict(symptoms_vector)
    rf_doenca = label_encoder.inverse_transform(rf_pred)[0]
    rf_confidence = rf_prob.max()  # Confiança do Random Forest

    # Previsão com Redes Neurais
    nn_prob = nn_model.predict(symptoms_vector)
    nn_pred = np.argmax(nn_prob, axis=1)
    nn_doenca = label_encoder.inverse_transform(nn_pred)[0]
    nn_confidence = np.max(nn_prob)  # Confiança das Redes Neurais

    # Encontrar o tratamento correspondente
    tratamento = tratamentos_df[tratamentos_df['Doença'] == knn_doenca]['Tratamentos'].values[0]

    return knn_doenca, rf_doenca, nn_doenca, tratamento, knn_confidence, rf_confidence, nn_confidence

# Função para mostrar a precisão e feedback
def mostrar_precisao():
    # KNN
    knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))
    # Random Forest
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    # Redes Neurais
    nn_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)[1]

    print(f"Precisão KNN: {knn_accuracy * 100:.2f}%")
    print(f"Precisão Random Forest: {rf_accuracy * 100:.2f}%")
    print(f"Precisão Redes Neurais: {nn_accuracy * 100:.2f}%")

# Função principal
def main():
    print("Olá, eu sou o assistente virtual de recomendações de medicamentos!")
    print("Por favor, descreva seus sintomas para eu te ajudar.")

    # Receber a descrição dos sintomas do paciente
    sintomas_input = input("Digite os sintomas que está sentindo: ")

    # Prever a doença e tratamento usando os três modelos
    knn_doenca, rf_doenca, nn_doenca, tratamento, knn_confidence, rf_confidence, nn_confidence = prever_doenca_tratamento(sintomas_input)

    print("\nResultados das Previsões:")
    print(f"Previsão KNN: {knn_doenca} (Confiança: {knn_confidence * 100:.2f}%)")
    print(f"Previsão Random Forest: {rf_doenca} (Confiança: {rf_confidence * 100:.2f}%)")
    print(f"Previsão Redes Neurais: {nn_doenca} (Confiança: {nn_confidence * 100:.2f}%)")
    print(f"Tratamento sugerido: {tratamento}")

    print("\nPrecisão dos Modelos:")
    mostrar_precisao()

if __name__ == "__main__":
    main()
