from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax
import numpy as np
from colorama import init, Fore
# Mariusz Buhaj, Łukasz Bosak.
# https://www.kurnik.pl/reversi/zasady.phtml
# pozostałe informacje w ReversiColorInstrukcja.txt
# Inicjalizacja colorama (żeby w powerschell były rożne kolory pionków, planszy i nagłówków)
init(autoreset=True)

class Reversi(TwoPlayerGame):

    """
    Klasa reprezentująca grę Reversi (Othello).

    Atrybuty:
    ----------
    players : list
        Lista graczy (AI lub człowiek).
    board : numpy.ndarray
        8x8 plansza gry, gdzie 0 oznacza puste pole, 1 oznacza gracza 1 (zielony), a 2 oznacza gracza 2 (niebieski).
    nplayer : int
        Numer aktualnego gracza (1 lub 2).
    current_player : int
        Aktualny gracz (1 lub 2).

    Metody:
    -------
    possible_moves():
        Zwraca listę możliwych ruchów dla bieżącego gracza.
    can_flip(x, y):
        Sprawdza, czy można przechwycić pionki w danym ruchu.
    check_direction(x, y, dx, dy):
        Sprawdza w jednym kierunku, czy w danym ruchu można przechwycić pionki przeciwnika.
    make_move(move):
        Wykonuje ruch, aktualizując planszę i przechwytując pionki.
    flip_discs(x, y):
        Przechwytuje pionki w wyniku ruchu.
    lose():
        Sprawdza, czy aktualny gracz przegrał (nie ma więcej możliwych ruchów).
    is_over():
        Sprawdza, czy gra się zakończyła (plansza pełna lub brak ruchów).
    show():
        Wyświetla aktualny stan planszy z kolorowymi pionkami.
    scoring():
        Zwraca wynik dla aktualnego gracza, licząc ilość jego pionków na planszy.
    """


    def __init__(self, players):

        """
        Inicjalizuje grę Reversi z podanymi graczami.

        Parametry:
        ----------
        players : list
            Lista graczy (AI lub człowiek).
        """

        self.players = players
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3][3], self.board[4][4] = 1, 1
        self.board[3][4], self.board[4][3] = 2, 2
        self.nplayer = 1
        self.current_player = 1

    def possible_moves(self):
        """
        Zwraca listę możliwych ruchów dla bieżącego gracza.

        Zwraca:
        -------
        list
            Lista współrzędnych (x, y) możliwych ruchów.
        """
        moves = []
        for x in range(8):
            for y in range(8):
                if self.board[x, y] == 0 and self.can_flip(x, y):
                    moves.append((x, y))
        return moves

    def can_flip(self, x, y):

        """
       Sprawdza, czy możliwe jest przechwycenie pionków przeciwnika w danym ruchu.

       Parametry:
       ----------
       x : int
           Wiersz planszy.
       y : int
           Kolumna planszy.

       Zwraca:
       -------
       bool
           True, jeśli można przechwycić pionki w danym ruchu, False w przeciwnym wypadku.
       """

        directions = [(1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]
        for dx, dy in directions:
            if self.check_direction(x, y, dx, dy):
                return True
        return False

    def check_direction(self, x, y, dx, dy):

        """
        Sprawdza w jednym kierunku, czy można przechwycić pionki przeciwnika.

        Parametry:
        ----------
        x : int
            Wiersz planszy.
        y : int
            Kolumna planszy.
        dx : int
            Kierunek w osi x.
        dy : int
            Kierunek w osi y.

        Zwraca:
        -------
        bool
            True, jeśli można przechwycić pionki w danym kierunku, False w przeciwnym wypadku.
        """

        nx, ny = x + dx, y + dy
        enemy_color = 3 - self.current_player
        if 0 <= nx < 8 and 0 <= ny < 8 and self.board[nx, ny] == enemy_color:
            while 0 <= nx < 8 and 0 <= ny < 8:
                nx += dx
                ny += dy
                if nx < 0 or ny < 0 or nx >= 8 or ny >= 8:
                    break
                if self.board[nx, ny] == 0:
                    break
                if self.board[nx, ny] == self.current_player:
                    return True
        return False

    def make_move(self, move):

        """
        Wykonuje ruch dla bieżącego gracza i przechwytuje pionki przeciwnika.

        Parametry:
        ----------
        move : tuple
            Współrzędne (x, y) ruchu.
        """

        x, y = move
        self.board[x, y] = self.current_player
        self.flip_discs(x, y)

    def flip_discs(self, x, y):
        """
        Przechwytuje pionki przeciwnika na planszy w wyniku ruchu.

        Parametry:
        ----------
        x : int
            Wiersz ruchu.
        y : int
            Kolumna ruchu.
        """

        directions = [(1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]
        for dx, dy in directions:
            if self.check_direction(x, y, dx, dy):
                nx, ny = x + dx, y + dy
                while self.board[nx, ny] != self.current_player:
                    self.board[nx, ny] = self.current_player
                    nx += dx
                    ny += dy

    def lose(self):
        """
        Sprawdza, czy bieżący gracz przegrał (brak możliwych ruchów).

        Zwraca:
        -------
        bool
            True, jeśli gracz przegrał, False w przeciwnym wypadku.
        """
        return not any(self.possible_moves())

    def is_over(self):
        """
        Sprawdza, czy gra się zakończyła (plansza pełna lub brak możliwych ruchów).

        Zwraca:
        -------
        bool
            True, jeśli gra się zakończyła, False w przeciwnym wypadku.
        """
        return (self.board != 0).all() or self.lose()

    def show(self):
        """
         Wyświetla aktualny stan planszy w terminalu z kolorowymi nagłówkami i pionkami.
         - Nagłówki wierszy i kolumn są czerwone.
         - Gracz 1 (zielony), Gracz 2 (niebieski), puste pola są białe.
         """
        print(Fore.RED + "  " + " ".join([str(i) for i in range(8)]))

        # Wyświetlamy każdą linię planszy z czerwonym numerem wiersza na początku
        for i, row in enumerate(self.board):
            print(Fore.RED + str(i), end=" ")  # Nagłówek wiersza
            for cell in row:
                if cell == 0:  # Puste pole
                    print(Fore.WHITE + "0", end=" ")
                elif cell == 1:  # Gracz 1 (zielony)
                    print(Fore.GREEN + "1", end=" ")
                elif cell == 2:  # Gracz 2 (żółty)
                    print(Fore.YELLOW + "2", end=" ")
            print()  # Przejście do nowej linii

    def scoring(self):
        """
        Zwraca wynik dla bieżącego gracza.

        Zwraca:
        -------
        int
            Liczba pionków bieżącego gracza na planszy.
        """
        return np.sum(self.board == self.current_player)


# Tworzymy algorytm sztucznej inteligencji
ai_algo = Negamax(4)

# Gracz 1 to AI, gracz 2 to człowiek
game = Reversi([AI_Player(ai_algo), Human_Player()])

if __name__ == "__main__":
    game.play()
