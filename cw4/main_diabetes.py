# Importowanie bibliotek
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#Autorzy: Bosak Lukasz, Buhaj Mariusz

# Klasyfikacja cukrzycy za pomocą drzewa decyzyjnego i SVM
#
# Skrypt służy do analizy i klasyfikacji cukrzycy na podstawie danych medycznych.
# Zbiór danych zawiera różne cechy, takie jak poziom glukozy, ciśnienie krwi, wiek, itp.
# Wynik (Outcome) wskazuje, czy osoba ma cukrzycę (1) czy nie (0).
#
# Przygotowanie środowiska:
# 1. Upewnij się, że masz zainstalowane następujące biblioteki:
#    - pandas
#    - scikit-learn
#    - matplotlib
#    - seaborn
# 2. Pobierz plik danych `diabetes.csv` i umieść go w folderze `data`.
# 3. Upewnij się, że posiadasz Python 3.8+ oraz odpowiednie środowisko uruchomieniowe.

def visualize_data(data):
    """
    Generuje przykładowe wizualizacje danych

    Parametry:
    - data (pd.DataFrame): Zbiór danych do wizualizacji

    Wizualizacje:
    - Heatmapa korelacji między cechami
    - Histogram wartości glukozy w zależności od wyniku (outcome)
    """

    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Heatmapa korelacji cech")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x="Glucose", hue="Outcome", kde=True, bins=30, palette="Set2")
    plt.title("Rozkład wartości glukozy w zależności od wyniku")
    plt.xlabel("Wartość glukozy")
    plt.ylabel("Liczba obserwacji")
    plt.legend(title="Outcome", labels=["No Diabetes", "Diabetes"])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def classify_new_data(model, scaler, feature_count):
    """
    Pozwala użytkownikowi na wprowadzenie danych do klasyfikacji
    w formacie ciągu liczb oddzielonych spacją (np. 5 116 70 28 0 30.1 0.349 25)
    i wykonuje predykcję za pomocą wybranego modelu

    Parametry:
    - model: Wytrenowany model klasyfikacyjny (np. drzewo decyzyjne, SVM)
    - scaler: Obiekt do skalowania danych (np. StandardScaler)
    - feature_count: Liczba cech w danych wejściowych

    Działanie:
    - Pobiera dane od użytkownika jako ciąg liczb oddzielonych spacją
    - Sprawdza poprawność liczby wprowadzonych cech
    - Skaluje dane zgodnie z modelem
    - Dokonuje predykcji klasy i wyświetla wynik
    """

    print("\nWprowadź dane do klasyfikacji (liczby oddzielone spacją):")
    user_input = input(f"Podaj {feature_count} wartości: ")
    input_data = list(map(float, user_input.split()))
    
    if len(input_data) != feature_count:
        print(f"Błąd: oczekiwano {feature_count} wartości, ale podano {len(input_data)}")
        return

    input_df = pd.DataFrame([input_data], columns=X_diabetes.columns)
    
    input_data_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_data_scaled)
    prediction_label = "Diabetes" if prediction[0] == 1 else "No Diabetes"
    print(f"\nPredykcja: {prediction_label}")

def main(dt_model, svm_model, scaler, feature_count):
    """
    Interaktywny interfejs użytkownika do klasyfikacji nowych danych.

    Parametry:
    - dt_model: Wytrenowany model drzewa decyzyjnego
    - svm_model: Wytrenowany model SVM
    - scaler: Obiekt do skalowania danych (np. StandardScaler)
    - feature_count: Liczba cech w danych wejściowych

    Działanie:
    - Umożliwia wybór modelu do klasyfikacji
    - Pozwala na wprowadzenie nowych danych do klasyfikacji
    - Obsługuje zamknięcie programu
    """
    while True:
        print("\nWybierz model do klasyfikacji:")
        print("1. Drzewo decyzyjne")
        print("2. SVM")
        print("3. Wyjdź")
        choice = input("Wybierz opcję (1/2/3): ")
        if choice == "1":
            classify_new_data(dt_model, scaler, feature_count)
        elif choice == "2":
            classify_new_data(svm_model, scaler, feature_count)
        elif choice == "3":
            print("Zakończono program")
            break
        else:
            print("Nieprawidłowy wybór, spróbuj ponownie.")

# Wczytanie zbioru danych
diabetes_data_path = 'data/diabetes.csv'
diabetes_data = pd.read_csv(diabetes_data_path)

# Podział na cechy i zmienną docelową
X_diabetes = diabetes_data.drop(columns=["Outcome"])
y_diabetes = diabetes_data["Outcome"]

# Podział na zbiór treningowy i testowy (80% treningowy, 20% testowy)
X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=77, stratify=y_diabetes
)

# Normalizacja danych
scaler = StandardScaler()
X_diabetes_train_scaled = scaler.fit_transform(X_diabetes_train)
X_diabetes_test_scaled = scaler.transform(X_diabetes_test)

# Drzewo decyzyjne
dt_diabetes_model = DecisionTreeClassifier(random_state=77)
dt_diabetes_model.fit(X_diabetes_train_scaled, y_diabetes_train)

# Predykcja i ewaluacja dla drzewa decyzyjnego
y_diabetes_pred_dt = dt_diabetes_model.predict(X_diabetes_test_scaled)
dt_diabetes_accuracy = accuracy_score(y_diabetes_test, y_diabetes_pred_dt)
dt_diabetes_report = classification_report(y_diabetes_test, y_diabetes_pred_dt, target_names=["No Diabetes", "Diabetes"])

print("Wyniki dla drzewa decyzyjnego:")
print(f"Dokładność: {dt_diabetes_accuracy}")
print("Raport klasyfikacji:")
print(dt_diabetes_report)

# SVM
svm_diabetes_model = SVC(kernel='rbf', random_state=77)
svm_diabetes_model.fit(X_diabetes_train_scaled, y_diabetes_train)

# Predykcja i ewaluacja dla SVM
y_diabetes_pred_svm = svm_diabetes_model.predict(X_diabetes_test_scaled)
svm_diabetes_accuracy = accuracy_score(y_diabetes_test, y_diabetes_pred_svm)
svm_diabetes_report = classification_report(y_diabetes_test, y_diabetes_pred_svm, target_names=["No Diabetes", "Diabetes"])

print("\nWyniki dla SVM:")
print(f"Dokładność: {svm_diabetes_accuracy}")
print("Raport klasyfikacji:")
print(svm_diabetes_report)

# Wizualizacja danych
visualize_data(diabetes_data)

# Interfejs terminalowy
main(dt_diabetes_model, svm_diabetes_model, scaler, feature_count=X_diabetes.shape[1])
