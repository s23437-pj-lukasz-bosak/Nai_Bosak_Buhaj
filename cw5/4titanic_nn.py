import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import os

"""
Autorzy: Mariusz Buhaj, Łukasz Bosak
pip install tensorflow matplotlib seaborn scikit-learn
python cifar10_classifier.py
classify - uruchamia scenariusz
train - wytrenowanie modelu i zapisanie na dysku
"""

def plot_loss_acc(history, model):
    """
    Wizualizacja strat i dokładności modelu w trakcie treningu

    Parameters:
    history : tensorflow.python.keras.callbacks.History
        Obiekt zawierający historię treningu modelu
    model : int
        Numer modelu (do celów opisowych na wykresie)
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'Model {model} - accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title(f'Model {model} - loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)
"""
Wczytanie danych o pasażerach Titanica z pliku CSV
"""
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
"""
Przygotowanie danych: usunięcie niepotrzebnych kolumn i uzupełnienie braków
"""

label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
"""
Kodowanie zmiennych kategorycznych
"""

X = data.drop('Survived', axis=1)
y = data['Survived']

scaler = StandardScaler()
X = scaler.fit_transform(X)
"""
Standaryzacja danych cech
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)
"""
Podział danych na zbiory treningowe i testowe
"""

model_path_1 = 'titanic_model_1.h5'
model_path_2 = 'titanic_model_2.h5'

if os.path.exists(model_path_1) or os.path.exists(model_path_2):
    choice = input("Modele istnieją na dysku. Chcesz je wytrenować ponownie czy dokonać klasyfikacji? (train/classify): ").strip().lower()
    if choice == 'classify':
        retrain = False
        if os.path.exists(model_path_1):
            model_1 = load_model(model_path_1)
            print("Model 1 został wczytany z dysku")
        if os.path.exists(model_path_2):
            model_2 = load_model(model_path_2)
            print("Model 2 został wczytany z dysku")

        features = input("Podaj wektor cech oddzielony spacjami (np. 22 1 0 3 0.0 30.0 ...): ")
        feature_vector = [float(x) for x in features.split()]

        if len(feature_vector) != X_train.shape[1]:
            print(f"Oczekiwano {X_train.shape[1]} cech, podano {len(feature_vector)}!")
        else:
            feature_vector = scaler.transform([feature_vector])
            if 'model_1' in locals():
                prediction_1 = model_1.predict(feature_vector)
                prediction_1 = 1 if prediction_1[0][0] >= 0.5 else 0
                print(f"Klasa modelu 1: {prediction_1}")
            if 'model_2' in locals():
                prediction_2 = model_2.predict(feature_vector)
                prediction_2 = 1 if prediction_2[0][0] >= 0.5 else 0
                print(f"Klasa modelu 2: {prediction_2}")
    else:
        retrain = True
else:
    retrain = True

if ('train' in locals()) or (retrain):
    model_1 = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    """
    Definicja i trening modelu 1
    """

    model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_1 = model_1.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_test, y_test))
    model_1.save(model_path_1)
    print(f"Model 1 został zapisany do {model_path_1}")

    model_2 = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    """
    Definicja i trening modelu 2
    """

    model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_2 = model_2.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_test, y_test))
    model_2.save(model_path_2)
    print(f"Model 2 został zapisany do {model_path_2}")

    test_loss_1, test_accuracy_1 = model_1.evaluate(X_test, y_test)
    print(f'Model 1 - test accuracy: {test_accuracy_1:.2f}, test loss: {test_loss_1:.2f}')

    test_loss_2, test_accuracy_2 = model_2.evaluate(X_test, y_test)
    print(f'Model 2 - test accuracy: {test_accuracy_2:.2f}, test loss: {test_loss_2:.2f}')

    plot_loss_acc(history_1, 1)
    plot_loss_acc(history_2, 2)
