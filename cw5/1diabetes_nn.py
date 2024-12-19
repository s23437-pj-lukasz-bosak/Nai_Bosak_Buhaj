from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, InputLayer
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
Autorzy: Mariusz Buhaj, Łukasz Bosak

instrukcje:
    - nalezy zainstalowac biblioteki: 
    pip install tensorflow pandas matplotlib scikit-learn
    - uruchomienie skryptu w terminalu:
    python diabetes_model.py
    - classify - uruchamia scenariusz
    - train - wytrenowanie modelu i zapisanie na dysku
"""



file_path = 'data/diabetes_dataset.csv'

data = pd.read_csv(file_path)
"""
Wczytanie danych z pliku CSV

file_path : str
    Ścieżka do pliku z danymi
data : pandas.DataFrame
    Zawiera dane o pacjentach, w tym wynik (Outcome) do predykcji
"""

X = data.drop(columns=['Outcome'])
y = data['Outcome']
"""
Podział danych na cechy (X) i etykiety (y)

X : pandas.DataFrame
    Dane wejściowe (cechy)
y : pandas.Series
    Wynik binarny (0 lub 1)
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37, stratify=y)
"""
Podział danych na zbiór treningowy i testowy

X_train : pandas.DataFrame
    Dane treningowe (cechy)
X_test : pandas.DataFrame
    Dane testowe (cechy)
y_train : pandas.Series
    Etykiety treningowe
y_test : pandas.Series
    Etykiety testowe
"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""
Standaryzacja danych

scaler : sklearn.preprocessing.StandardScaler
    Służy do normalizacji danych wejściowych
"""

model_path = 'diabetes_model.h5'
"""
Ścieżka do pliku, w którym zapisany jest model
"""

if os.path.exists(model_path):
    choice = input("Model istnieje na dysku. Chcesz go wytrenować czy dokonać klasyfikacji? (train/classify): ").strip().lower()

    if choice == 'classify':
        model = load_model(model_path)
        print("Model został wczytany z dysku")
        """
        Wczytanie zapisanego modelu z dysku

        model : tensorflow.keras.models.Sequential
            Wczytany model sieci neuronowej
        """

        features = input("\nPodaj wektor cech oddzielony spacjami (np. 23 0.93 53.5 923.21 ...): ")
        feature_vector = [float(x) for x in features.split()]

        if len(feature_vector) != X_train.shape[1]:
            print(f"Oczekiwano {X_train.shape[1]} cech, podano {len(feature_vector)}!")
        else:
            feature_vector = scaler.transform([feature_vector])
            prediction = model.predict(feature_vector)
            """
            Dokonanie predykcji dla podanego wektora cech

            feature_vector : list
                Znormalizowany wektor cech
            prediction : int
                Wynik predykcji (0 lub 1)
            """
            prediction = 1 if prediction[0][0] >= 0.5 else 0
            print(f"Klasa: {prediction}")
    else:
        print("Ponowne trenowanie modelu...")
        retrain = True
else:
    retrain = True

if 'retrain' in locals() and retrain:
    model = Sequential([
        InputLayer(shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    """
    Tworzenie nowego modelu sieci neuronowej

    model : tensorflow.keras.models.Sequential
        Nowy model sieci neuronowej z warstwami Dense
    """

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)
    """
    Trenowanie modelu na danych treningowych

    history : tensorflow.python.keras.callbacks.History
        Historia treningu modelu
    """

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {test_loss}, test accuracy: {test_accuracy}')
    """
    Ewaluacja modelu na zbiorze testowym

    test_loss : float
        Funkcja kosztu na zbiorze testowym
    test_accuracy : float
        Dokładność na zbiorze testowym
    """

    model.save(model_path)
    print(f"Model został zapisany do {model_path}")
    """
    Zapis modelu do pliku.

    model_path : str
        Ścieżka do pliku, w którym zapisano model
    """

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
