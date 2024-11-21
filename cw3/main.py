import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import requests

# Autorzy: Bosak Łukasz, Buhaj Mariusz

"""
Opis problemu:
---------------
Celem programu jest generowanie rekomendacji filmowych na podstawie ocen użytkowników oraz klasteryzacji danych. 
Problemem, który napotkano, było brak rekomendacji dla niektórych użytkowników. 
Przyczyną mogło być:
1. Brak wystarczających danych w klastrze użytkownika.
2. Użytkownik był jedynym członkiem swojego klastra, co uniemożliwiało generowanie rekomendacji na podstawie innych 
   użytkowników.
3. Wszystkie filmy w klastrze zostały już ocenione przez użytkownika.


Instrukcja użycia:
------------------
1. Przygotowanie danych:
   - Plik Excel powinien zawierać dane w formacie:
     - Pierwsza kolumna: Nazwy użytkowników.
     - Kolejne kolumny: Filmy i ich oceny (oceny jako liczby, puste miejsca są dozwolone).

2. Uruchomienie programu:
   - Skrypt automatycznie wczytuje dane z pliku Excel (`data_path`) i tworzy macierz użytkownik-film.
   - Dane są normalizowane, a następnie wykonywana jest klasteryzacja hierarchiczna.

3. Wybór użytkownika:
   - Po zakończeniu klasteryzacji wyświetlana jest lista użytkowników.
   - Wprowadź numer użytkownika, aby zobaczyć rekomendacje dla niego.

4. Wyniki:
   - Program wyświetli rekomendacje (najlepiej oceniane filmy w klastrze, których użytkownik jeszcze nie ocenił) 
     oraz antyrekomendacje (najgorzej oceniane filmy w klastrze).
   - Jeśli nie ma dostępnych rekomendacji w klastrze, użytkownik nie otrzyma żadnych wyników.

Uwagi:
------
- Program wymaga zainstalowanych bibliotek: `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `requests`.
- Klucz API do OMDB (`omdb_api_key`) jest opcjonalny i służy do pobierania szczegółowych informacji o filmach.
"""
def read_and_process_data(data_path):
    """
    Wczytuje i przetwarza dane dotyczące ocen filmów.

    Args:
        data_path (str): Ścieżka do pliku Excel zawierającego dane o filmach

    Returns:
        pd.DataFrame: Oczyszczony zbiór danych z kolumnami ['User', 'Movie', 'Rating']
    """
    data = pd.read_excel(data_path, sheet_name='List1')
    num_users = data.shape[0]

    movies_corrected = []
    ratings_corrected = []
    users_corrected = []

    for i, movie in enumerate(data.iloc[:, 1:].stack().values[::2]):
        user_index = i % num_users
        user = data.iloc[user_index, 0]
        users_corrected.append(user)
        movies_corrected.append(movie)
        ratings_corrected.append(data.iloc[:, 1:].stack().values[1::2][i])

    corrected_data = pd.DataFrame({
        'User': users_corrected,
        'Movie': movies_corrected,
        'Rating': ratings_corrected
    }).drop_duplicates()

    corrected_data['Rating'] = pd.to_numeric(corrected_data['Rating'], errors='coerce')
    return corrected_data

def elbow_method_plot(scaled_data, max_clusters=10):
    """
    Generuje wykres metody łokcia

    Args:
        scaled_data (array): Znormalizowana macierz użytkownik-film
        max_clusters (int): Maksymalna liczba klastrów do przetestowania

    Returns:
        None
    """
    wcss = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
    plt.title('Metoda łokcia dla optymalnej liczby klastrów')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputFiles/metoda_lokcia.png')
    plt.show()

def hierarchical_clustering(scaled_data, metric, method='average', num_clusters=5):
    """
    Wykonuje hierarchiczną klasteryzację danych

    Args:
        scaled_data (array): Znormalizowana macierz użytkownik-film
        metric (str): Metryka odległości do użycia (np. 'euclidean', 'cityblock', 'cosine')
        method (str): Metoda łączenia (np. 'ward', 'complete', 'average', 'single')
        num_clusters (int): Liczba klastrów do wygenerowania

    Returns:
        dict: Wyniki klasteryzacji zawierające przypisania klastrów i silhouette score
    """
    linkage_matrix = linkage(scaled_data, method=method, metric=metric)
    cluster_assignments = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    silhouette_avg = silhouette_score(scaled_data, cluster_assignments, metric='euclidean')
    return {'Clusters': cluster_assignments, 'Silhouette Score': silhouette_avg}

def plot_dendrogram(scaled_data, metric, method='average'):
    """
    Wyświetla dendrogram dla hierarchicznej klasteryzacji

    Args:
        scaled_data (array): Znormalizowana macierz użytkownik-film
        metric (str): Metryka odległości do użycia (np. 'euclidean', 'cityblock', 'cosine')
        method (str): Metoda łączenia (np. 'ward', 'average', 'complete', 'single')

    Returns:
        None
    """
    linkage_matrix = linkage(scaled_data, method=method, metric=metric)
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=user_movie_matrix.index, leaf_rotation=90, leaf_font_size=10)
    plt.title(f"Dendrogram (Odległość: {metric})")
    plt.xlabel('Użytkownik')
    plt.ylabel('Odległość')
    plt.tight_layout()
    plt.savefig(f'outputFiles/dendogram_{method}.png')
    plt.show()

def generate_cluster_recommendations(user, cluster_data, corrected_data, n_recommendations=5, n_antirecommendations=5):
    """
    Generuje rekomendacje i antyrekomendacje dla użytkownika na podstawie klastrów

    Args:
        user (str): Użytkownik, dla którego generowane są rekomendacje
        cluster_data (pd.DataFrame): DataFrame z przypisaniem użytkowników do klastrów
        corrected_data (pd.DataFrame): Oryginalne dane z kolumnami ['User', 'Movie', 'Rating']
        n_recommendations (int): Liczba rekomendacji do wygenerowania
        n_antirecommendations (int): Liczba antyrekomendacji do wygenerowania

    Returns:
        dict: Zawiera rekomendacje i antyrekomendacje jako pd.Series
    """
    user_cluster = cluster_data.loc[cluster_data['User'] == user, 'Cluster'].values[0]
    users_in_cluster = cluster_data.loc[cluster_data['Cluster'] == user_cluster, 'User']
    cluster_ratings = corrected_data[corrected_data['User'].isin(users_in_cluster)]
    avg_ratings = cluster_ratings.groupby('Movie')['Rating'].mean()
    user_movies = corrected_data[corrected_data['User'] == user]['Movie']
    recommendations = avg_ratings.drop(user_movies, errors='ignore').sort_values(ascending=False).head(n_recommendations)
    antirecommendations = avg_ratings.drop(user_movies, errors='ignore').sort_values().head(n_antirecommendations)
    return {"Recommendations": recommendations, "Anti-Recommendations": antirecommendations}

def fetch_movie_info(movie_title, api_key):
    """
    Pobiera informacje o filmie z OMDB API

    Args:
        movie_title (str): Tytuł filmu
        api_key (str): Klucz API do OMDB

    Returns:
        dict: Informacje o filmie (np. gatunek, fabuła) lub komunikat błędu
    """
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['Response'] == 'True':
            return {
                'Title': data.get('Title', 'N/A'),
                'Year': data.get('Year', 'N/A'),
                'Genre': data.get('Genre', 'N/A'),
                'Director': data.get('Director', 'N/A'),
                'Plot': data.get('Plot', 'N/A'),
                'IMDB Rating': data.get('imdbRating', 'N/A')
            }
        else:
            return {'Error': data.get('Error', 'Nieznany błąd')}
    else:
        return {'Error': 'Błąd połączenia z OMDB API'}

def additional_movie_info(recommendations, api_key):
    """
    Wzbogaca rekomendacje o dodatkowe informacje z OMDB API

    Args:
        recommendations (pd.Series): Seria z tytułami rekomendowanych filmów
        api_key (str): Klucz API do OMDB

    Returns:
        pd.DataFrame: DataFrame z dodatkowymi informacjami o filmach
    """
    enriched_data = []
    for movie in recommendations.index:
        movie_info = fetch_movie_info(movie, api_key)
        enriched_data.append(movie_info)
    return pd.DataFrame(enriched_data)


# Wczytanie danych
data_path = 'data/bazafilmow.xlsx'
data = read_and_process_data(data_path)

# Tworzenie macierzy użytkownik-film
user_movie_matrix = data.pivot_table(index='User', columns='Movie', values='Rating', fill_value=0)

# Konwersja typu anzw kolumn
user_movie_matrix.columns = user_movie_matrix.columns.astype(str)

# Normalizacja danych
scaler = StandardScaler()
user_movie_scaled = scaler.fit_transform(user_movie_matrix)

# Metoda łokcia do determinacji liczby klastrów
elbow_method_plot(user_movie_scaled)

# Klasteryzacja hierarchiczna (odległość Euklidesowa)
hierarchical_euclidean_results = hierarchical_clustering(user_movie_scaled, metric='euclidean', method='ward', num_clusters=5)

# Klasteryzacja hierarchiczna (odłegłość Manhattan)
hierarchical_manhattan_results = hierarchical_clustering(user_movie_scaled, metric='cityblock', method='average', num_clusters=5)

# Wyświetlenie wyników
print("Hierarchical Clustering Results (Euclidean):", hierarchical_euclidean_results)
print("Hierarchical Clustering Results (Manhattan):", hierarchical_manhattan_results)

# Dendrogram dla odległości Euklidesowej
plot_dendrogram(user_movie_scaled, metric='euclidean', method='ward')

# Dendrogram dla odległości Manhattan
plot_dendrogram(user_movie_scaled, metric='cityblock', method='average')

# Tworzenie df z klastrami użytkowników
user_clusters = pd.DataFrame({
    'User': user_movie_matrix.index,
    'Cluster': hierarchical_euclidean_results['Clusters']
})

# Klucz do API OMDB
omdb_api_key = "3cd59c6c"

# Lista dostępnych użytkowników
print("Dostępni użytkownicy:")
for idx, user in enumerate(user_clusters['User']):
    print(f"{idx + 1}. {user}")

# Wybór użytkownika z wiersza poleceń
selected_user_idx = int(input("Wybierz numer użytkownika, dla którego chcesz zobaczyć rekomendacje: ")) - 1

# Walidacja wyboru
if 0 <= selected_user_idx < len(user_clusters):
    selected_user = user_clusters['User'].iloc[selected_user_idx]
    print(f"\nWybrano użytkownika: {selected_user}")

    # Generowanie wyników dla wybranego użytkownika
    results = generate_cluster_recommendations(selected_user, user_clusters, data, n_recommendations=5, n_antirecommendations=5)

    # Wyświetlenie wyników
    print("\nRekomendacje:")
    print(results["Recommendations"])
    print("\nAntyrekomendacje:")
    print(results["Anti-Recommendations"])

    # Pobieranie dodatkowych informacji o filmach
    detailed_recommendations = additional_movie_info(results["Recommendations"], omdb_api_key)
    detailed_antirecommendations = additional_movie_info(results["Anti-Recommendations"], omdb_api_key)

    print("\nSzczegóły rekomendacji:")
    print(detailed_recommendations)
    print("\nSzczegóły antyrekomendacji:")
    print(detailed_antirecommendations)
else:
    print("Nieprawidłowy wybór. Upewnij się, że podano poprawny numer użytkownika.")
