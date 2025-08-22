
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[['title', 'genres']].dropna()
    df['genres'] = df['genres'].str.lower().str.replace('|', ' ', regex=False)
    df['title'] = df['title'].str.strip().str.lower()
    return df

movies = load_data()

def simple_genre_recommendation(user_genre, num_recommendations=5):
    user_genre = user_genre.strip().lower()
    matching = movies[movies['genres'].str.contains(user_genre, na=False)]
    if matching.empty:
        return f"No movies found for genre: {user_genre}"
    return matching.sample(n=min(num_recommendations, len(matching)))['title'].tolist()

def polished_genre_recommendation(user_genre, num_recommendations=5):
    user_genre = user_genre.strip().lower()
    matching = movies[movies['genres'].str.contains(user_genre, na=False)]
    if matching.empty:
        return f"Sorry, we couldn't find any movies in the '{user_genre}' genre."
    return matching.sample(n=min(num_recommendations, len(matching)))['title'].tolist()

@st.cache_resource
def setup_tfidf_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])
    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(tfidf_matrix)
    return tfidf_matrix, nn

tfidf_matrix, nn_model = setup_tfidf_model(movies)
title_index = pd.Series(movies.index, index=movies['title'])

def recommend_by_title(title, n_recommendations=5):
    title = title.lower()
    if title not in title_index:
        return "Movie not found. Please check the exact title with year (e.g., 'The Dark Knight (2008)')."
    idx = title_index[title]
    tfidf_vector = tfidf_matrix[idx]
    distances, indices_nn = nn_model.kneighbors(tfidf_vector, n_neighbors=n_recommendations + 1)
    recommended_indices = indices_nn[0][1:]
    return movies['title'].iloc[recommended_indices].tolist()

st.title("🎬 Movie Recommendation System")
st.sidebar.header("Choose Recommendation Type")

option = st.sidebar.radio(
    "Select an option:",
    (
        "🎯 Genre-Based Recommendation (Simple)",
        "🌟 Genre-Based Recommendation (Polished)",
        "🎥 Movie Title-Based Recommendation"
    )
)

if option == "🎯 Genre-Based Recommendation (Simple)":
    genre = st.text_input("Enter a genre (e.g., Action, Comedy, Drama):")
    if st.button("Get Recommendations"):
        result = simple_genre_recommendation(genre)
        st.subheader(f"Recommended {genre.title()} Movies:")
        if isinstance(result, list):
            for i, movie in enumerate(result, 1):
                st.write(f"{i}. {movie}")
        else:
            st.warning(result)

elif option == "🌟 Genre-Based Recommendation (Polished)":
    genre = st.text_input("Please enter a movie genre you like (e.g., Action, Comedy, Drama):")
    if st.button("Show Recommendations"):
        result = polished_genre_recommendation(genre)
        st.subheader(f"Top {genre.title()} Movie Picks:")
        if isinstance(result, list):
            for i, movie in enumerate(result, 1):
                st.write(f"{i}. {movie}")
        else:
            st.error(result)

elif option == "🎥 Movie Title-Based Recommendation":
    title = st.text_input("Enter the exact movie title (e.g., 'The Dark Knight (2008)'):", key="title_input")
    if st.button("Find Similar Movies"):
        result = recommend_by_title(title)
        st.subheader(f"Movies similar to '{title.title()}':")
        if isinstance(result, list):
            for i, movie in enumerate(result, 1):
                st.write(f"{i}. {movie}")
        else:
            st.error(result)
