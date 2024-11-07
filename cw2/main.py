import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# autorzy: Mariusz Buhaj, Łukasz Bosak

def stworz_system_finansowy():
    """
    Tworzy system finansowy oparty na logice rozmytej, który ocenia stan finansowy gospodarstwa domowego
    na podstawie trzech wejściowych zmiennych: dochody, wydatki i oszczędności.

    Zmienne wejściowe:
    - dochody: zakres od 0 do 20 000 PLN
    - wydatki: zakres od 0 do 20 000 PLN
    - oszczędności: zakres od 0 do 100 000 PLN

    Zmienna wyjściowa:
    - stan_finansowy: zakres od 0 do 100, gdzie 0 oznacza zły stan finansowy, a 100 oznacza bardzo dobry stan finansowy.

    Zwraca:
    - Obiekt ControlSystemSimulation gotowy do wprowadzania danych wejściowych i obliczania wyników.
    """

    # Wejścia: dochody, wydatki, oszczędności
    dochody = ctrl.Antecedent(np.arange(0, 20001, 1), 'dochody')
    wydatki = ctrl.Antecedent(np.arange(0, 20001, 1), 'wydatki')
    oszczednosci = ctrl.Antecedent(np.arange(0, 100001, 1), 'oszczednosci')

    # Wyjście: stan finansowy
    stan_finansowy = ctrl.Consequent(np.arange(0, 101, 1), 'stan_finansowy')

    # Funkcje przynależności dla dochodów (niskie, średnie, wysokie)
    dochody['niskie'] = fuzz.trimf(dochody.universe, [0, 0, 12000])
    dochody['srednie'] = fuzz.trimf(dochody.universe, [8000, 12000, 16000])
    dochody['wysokie'] = fuzz.trimf(dochody.universe, [12000, 20000, 20000])

    # Funkcje przynależności dla wydatków (niskie, średnie, wysokie)
    wydatki['niskie'] = fuzz.trimf(wydatki.universe, [0, 0, 10000])
    wydatki['srednie'] = fuzz.trimf(wydatki.universe, [5000, 10000, 15000])
    wydatki['wysokie'] = fuzz.trimf(wydatki.universe, [10000, 20000, 20000])

    # Funkcje przynależności dla oszczędności (mało, średnio, dużo)
    oszczednosci['malo'] = fuzz.trimf(oszczednosci.universe, [0, 0, 30000])
    oszczednosci['srednio'] = fuzz.trimf(oszczednosci.universe, [20000, 50000, 80000])
    oszczednosci['duzo'] = fuzz.trimf(oszczednosci.universe, [50000, 100000, 100000])

    # Funkcje przynależności dla stanu finansowego (zły, średni, dobry)
    stan_finansowy['zly'] = fuzz.trimf(stan_finansowy.universe, [0, 0, 40])
    stan_finansowy['sredni'] = fuzz.trimf(stan_finansowy.universe, [30, 50, 70])
    stan_finansowy['dobry'] = fuzz.trimf(stan_finansowy.universe, [60, 100, 100])

    # Reguły logiki rozmytej
    rule1 = ctrl.Rule(dochody['niskie'] & wydatki['wysokie'] & oszczednosci['malo'], stan_finansowy['zly'])
    rule2 = ctrl.Rule(dochody['srednie'] & wydatki['srednie'] & oszczednosci['srednio'], stan_finansowy['sredni'])
    rule3 = ctrl.Rule(dochody['wysokie'] & wydatki['niskie'] & oszczednosci['duzo'], stan_finansowy['dobry'])
    rule4 = ctrl.Rule(dochody['niskie'] & wydatki['srednie'] & oszczednosci['malo'], stan_finansowy['zly'])
    rule5 = ctrl.Rule(dochody['wysokie'] & wydatki['wysokie'] & oszczednosci['srednio'], stan_finansowy['sredni'])

    # Dodanie reguły domyślnej (fallback)
    rule_default = ctrl.Rule(dochody['niskie'] | wydatki['niskie'] | oszczednosci['malo'], stan_finansowy['zly'])

    # Tworzenie systemu sterowania
    system_finansowy = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule_default])
    symulacja_finansowa = ctrl.ControlSystemSimulation(system_finansowy)

    return symulacja_finansowa


def oblicz_stan_finansowy(symulacja_finansowa, dochody, wydatki, oszczednosci):
    """
    Oblicza stan finansowy gospodarstwa domowego na podstawie podanych danych wejściowych.

    Argumenty:
    - symulacja_finansowa: obiekt ControlSystemSimulation, który został zwrócony z funkcji stworz_system_finansowy
    - dochody: miesięczne dochody gospodarstwa domowego (od 0 do 20 000 PLN)
    - wydatki: miesięczne wydatki gospodarstwa domowego (od 0 do 20 000 PLN)
    - oszczednosci: oszczędności gospodarstwa domowego (od 0 do 100 000 PLN)

    Zwraca:
    - Wartość zmiennej 'stan_finansowy', która określa stan finansowy gospodarstwa (od 0 do 100).
    """

    # Przypisanie danych wejściowych
    symulacja_finansowa.input['dochody'] = dochody
    symulacja_finansowa.input['wydatki'] = wydatki
    symulacja_finansowa.input['oszczednosci'] = oszczednosci

    # Obliczenie wyniku
    symulacja_finansowa.compute()

    # Sprawdzenie, czy wynik został obliczony
    if 'stan_finansowy' in symulacja_finansowa.output:
        return symulacja_finansowa.output['stan_finansowy']
    else:
        print(f"Dochody: {dochody}, Wydatki: {wydatki}, Oszczędności: {oszczednosci}")
        raise ValueError("Wynik 'stan_finansowy' nie został obliczony. Sprawdź dane wejściowe lub reguły.")


def opisz_stan_finansowy(stan):
    """
    Opisuje stan finansowy gospodarstwa domowego na podstawie wyniku symulacji.
    Zwraca opis tekstowy w zależności od wartości wyniku.

    Argumenty:
    - stan: wynik symulacji, liczba w zakresie od 0 do 100.

    Zwraca:
    - Tekstowy opis stanu finansowego.
    """
    if stan < 30:
        return "Zły stan finansowy gospodarstwa domowego."
    elif 30 <= stan < 70:
        return "Średni stan finansowy gospodarstwa domowego."
    else:
        return "Dobry stan finansowy gospodarstwa domowego."


# Główna część programu
if __name__ == "__main__":
    # Tworzenie systemu finansowego
    symulacja = stworz_system_finansowy()

    # Pobieranie danych od użytkownika
    dochody_przyklad = float(input("Podaj miesięczne dochody (PLN): "))
    wydatki_przyklad = float(input("Podaj miesięczne wydatki (PLN): "))
    oszczednosci_przyklad = float(input("Podaj wartość oszczędności (PLN): "))

    # Obliczenie stanu finansowego
    stan = oblicz_stan_finansowy(symulacja, dochody_przyklad, wydatki_przyklad, oszczednosci_przyklad)

    # Wyświetlenie wyniku z dwoma miejscami po przecinku oraz opis stanu
    print(f"Stan finansowy gospodarstwa domowego: {stan:.2f}")
    print(opisz_stan_finansowy(stan))