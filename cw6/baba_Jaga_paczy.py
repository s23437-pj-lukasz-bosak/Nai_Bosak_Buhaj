import cv2

# Autorzy: Mariusz Buhaj, Łukasz Bosak
#
# aby uruchomic nalezy zainstalowac biblioteke opencv (do przetwarzania obrazu i analizy wideo):
#     pip install opencv-python
#
# Zielony celownik: Rysowany, gdy twarz jest nieruchoma.
# Czerwony celownik: Pojawia się, gdy twarz się poruszy.
# Kod stale monitoruje ruch twarzy i dynamicznie zmienia kolor celownika.
# Zakończenie programu po naciśnięciu klawisza 'q'

# Inicjalizacja kamery, uruchamia kamerę 0 czyli domyślną.
cap = cv2.VideoCapture(0)

# Wczytanie klasyfikatora detekcji twarzy - Ładuje gotowy model detekcji twarzy oparty na algorytmie Haar Cascade.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Zmienna do przechowywania pozycji twarzy
prev_face_position = None

#Pętla przetwarzająca klatki wideo.
while True:
    ret, frame = cap.read() #pobiera klatke z kamery

    # Konwersja obrazu na skalę szarości - gdyż detekcja twarzy jest szybsza w odcieniach szarosci.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywanie twarzy
    #       detectMultiScale: Wykrywa twarze w obrazie.
    #       scaleFactor=1.1: Skaluje obraz w poszukiwaniu twarzy o różnych rozmiarach.
    #       minNeighbors=5: Określa, ile sąsiadujących prostokątów musi być wykrytych, aby uznać, że to twarz.

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Sprawdzanie pozycji twarzy
    if len(faces) > 0:
        x, y, w, h = faces[0]
        current_face_position = (x, y, w, h) # zmienna ta przechowuje pozycje twarzy.

        # Domyślny kolor celownika (zielony)
        color = (0, 255, 0)

        if prev_face_position is not None:
            # Obliczanie różnicy pozycji twarzy
            delta_x = abs(prev_face_position[0] - current_face_position[0])
            delta_y = abs(prev_face_position[1] - current_face_position[1])

            # Jeśli ruch został wykryty, zmień kolor na czerwony ( ruch wykryty jesli delta_x/y wieksza niz 5 pikseli)
            if delta_x > 5 or delta_y > 5:
                color = (0, 0, 255)

        # Obliczanie środka twarzy i promienia okręgu
        center = (x + w // 2, y + h // 2)
        radius = min(w, h) // 2

        # Rysowanie okręgu
        cv2.circle(frame, center, radius, color, 5)

        # Obliczanie rozmiaru krzyża w okręgu
        cross_size = min(w, h) // 4

        # Rysowanie pionowej linii krzyża
        cv2.line(frame, (center[0], center[1] - cross_size),
                 (center[0], center[1] + cross_size), color, 5)

        # Rysowanie poziomej linii krzyża
        cv2.line(frame, (center[0] - cross_size, center[1]),
                 (center[0] + cross_size, center[1]), color, 5)

        # Aktualizacja pozycji twarzy
        prev_face_position = current_face_position

    # Wyświetlanie obrazu z naniesionym celownikiem
    cv2.imshow('strzelanie', frame)

    # Zakończenie programu po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów
cap.release()
cv2.destroyAllWindows()
