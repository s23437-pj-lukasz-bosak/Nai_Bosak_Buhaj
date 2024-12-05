# Importowanie bibliotek
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

#Autorzy: Bosak Lukasz, Buhaj Mariusz

# Klasyfikacja sygnałów sonarowych za pomocą drzewa decyzyjnego i SVM
#
# Skrypt analizuje dane sonarowe, które wskazują, czy dany sygnał pochodzi od skały (R - Rock) czy miny (M - Mine).
# Zbiór danych zawiera 60 cech numerycznych dla każdego sygnału. Klasa końcowa jest zakodowana jako 'R' (0) lub 'M' (1).
#
# Przygotowanie środowiska:
# 1. Zainstaluj wymagane biblioteki:
#    - pandas
#    - matplotlib
#    - seaborn
#    - scikit-learn
# 2. Pobierz plik danych `sonar.csv` i umieść go w folderze `data`.
#    Plik można znaleźć na stronie UCI Machine Learning Repository (Sonar Dataset).
# 3. Upewnij się, że posiadasz Python w wersji 3.8 lub nowszej.
#
# Działanie skryptu:
# - Dzieli dane na cechy i etykiety oraz na zbiory treningowy i testowy.
# - Normalizuje cechy za pomocą `StandardScaler`.
# - Trenuje dwa modele: drzewo decyzyjne i SVM.
# - Przeprowadza ewaluację każdego modelu, generując raporty klasyfikacji.
# - Wizualizuje dane za pomocą heatmapy korelacji oraz wykresu dwóch cech.
# - Umożliwia użytkownikowi interaktywną klasyfikację nowych danych.

def visualize_data(data, label_column):
    """
    Generuje przykładowe wizualizacje danych

    Parametry:
    - data (pd.DataFrame): Zbiór danych do wizualizacji
    - label_column (str): Nazwa kolumny z etykietami klasy

    Wizualizacje:
    - Heatmapa korelacji między cechami
    - Rozkład pierwszych dwóch cech z podziałem na klasy
    """

    features_only = data.drop(columns=[label_column])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(features_only.corr(), cmap="coolwarm", annot=False, vmin=-1, vmax=1)
    plt.title("Heatmapa korelacji cech")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    for label, color in zip(data[label_column].unique(), ['red', 'blue']):
        subset = data[data[label_column] == label]
        plt.scatter(subset[0], subset[1], label=label, alpha=0.6, edgecolors='k')
    plt.title("Wizualizacja dwóch pierwszych zmiennych z zestawu 'sonar'")
    plt.xlabel("Zmienna 1")
    plt.ylabel("Zmienna 2")
    plt.legend(title="Class")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def classify_new_data(model, scaler, feature_count):
    """
    Pozwala użytkownikowi na wprowadzenie danych do klasyfikacji
    w formacie ciągu liczb oddzielonych spacją (np. 0.2 3 5 0.3 594)
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

    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    prediction_label = "Mine" if prediction[0] == 1 else "Rock"
    print(f"\nPredykcja: {prediction_label}")


def main(dt_model, svm_model, scaler, feature_count):
    """
    Interaktywny interfejs użytkownika do klasyfikacji nowych danych

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
            print("Nieprawidłowy wybór, sprobój ponownie")

# Wczytanie zbioru danych
file_path = 'data/sonar.csv'
sonar_data = pd.read_csv(file_path, header=None)

# Podział na cechy i etykiety
X = sonar_data.iloc[:, :-1]
y = sonar_data.iloc[:, -1]

# Kodowanie etykiet (M -> 1, R -> 0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=77, stratify=y_encoded
)

# Normalizacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Drzewo decyzyjne
dt_model = DecisionTreeClassifier(random_state=77)
dt_model.fit(X_train_scaled, y_train)

# Ewaluacja drzewa decyzyjnego
y_pred_dt = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_report = classification_report(y_test, y_pred_dt, target_names=label_encoder.classes_)

print("Wyniki dla drzewa decyzyjnego:")
print(f"Dokładność: {dt_accuracy}")
print("Raport klasyfikacji:")
print(dt_report)

# SVM
svm_model = SVC(kernel='rbf', random_state=77)
svm_model.fit(X_train_scaled, y_train)

# Ewaluacja SVM
y_pred_svm = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_)

print("\nWyniki dla SVM:")
print(f"Dokładność: {svm_accuracy}")
print("Raport klasyfikacji:")
print(svm_report)

# Wizualizacja danych
visualize_data(sonar_data, label_column=60)

# Interaktywny interfejs użytkownika
main(dt_model, svm_model, scaler, feature_count=X.shape[1])
