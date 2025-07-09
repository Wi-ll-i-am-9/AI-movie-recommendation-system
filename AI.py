import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init, Fore
import time


init(autoreset=True)

def load_data(file_path='imdb_top_100.csv'):
    try:
        df = pd.read_csv(file_path)
        df["combined_features"] = df["Genre"].fillna("") + " " + df["Overview"].fillna("")
        return df
    except FileNotFoundError:
        print(Fore.RED + f"Error: The file {file_path} was not found.")
        exit()

movies_df = load_data()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def list_genres(df):
    return sorted(set(genre.strip() for sublist in df["Genre"].dropna().str.split(", ") for genre in sublist))

genres = list_genres(movies_df)

def recommend_movies(genre=None, mood=None, rating=None, top_n=5):
    filtered_df = movies_df
    if genre:
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(genre, case=False, na=False)]
    if rating:
        filtered_df = filtered_df[filtered_df['IMDB_Rating'] >= rating]

    filtered_df = filtered_df.sample(frac=1).reset

