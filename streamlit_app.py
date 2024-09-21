import streamlit as st
import joblib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit.components.v1 as components
import os

# Load pre-trained models and preprocessors
a_scaler = joblib.load('a_scaler.pkl')
kmeans = joblib.load('kmeans.pkl')
ewd_lo_disc = joblib.load('ewd_lo_disc.pkl')
rf = joblib.load('song_classification_model.pkl')

# Spotify API Credentials
load_dotenv()
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                                           client_secret=SPOTIPY_CLIENT_SECRET))


# Streamlit app layout
st.set_page_config(layout="wide",page_title='DecadeTune')
adsense_code = """
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-4217328231836897"
     crossorigin="anonymous"></script>
"""
components.html(adsense_code, height=0)
st.title("🎶 DecadeTune, A Song Classifier for Decade Groups")
st.subheader("Enter a Spotify song URL, and I'll tell you if it sounds like a hit from the 50s-60s, 70s-80s, 90s-2000s, or 2010s-2020s:")

# Input field for Spotify song URL
song_id = st.text_input("Enter a Spotify song URL (e.g https://open.spotify.com/track/2rWny2VBj64zAx9q5VvPkB?si=75f33297b7ac40b9):")

# Function to get song features and album cover from Spotify API
def get_song_features(song_id):
    try:
        track_info = sp.track(song_id)
        features = sp.audio_features(song_id)[0]

        if features:
            album_cover_url = track_info['album']['images'][0]['url']
            song_data = {
                'Track Name': track_info['name'],
                'Artist': track_info['artists'][0]['name'],
                'Album Cover': album_cover_url,
                'Duration_ms': features['duration_ms'],
                'Danceability': features['danceability'],
                'Energy': features['energy'],
                'Key': features['key'],
                'Loudness': features['loudness'],
                'Mode': features['mode'],
                'Speechiness': features['speechiness'],
                'Acousticness': features['acousticness'],
                'Instrumentalness': features['instrumentalness'],
                'Liveness': features['liveness'],
                'Valence': features['valence'],
                'Time_Signature': features['time_signature'],
                'Tempo': features['tempo']
            }
            return pd.DataFrame([song_data])
        else:
            st.error("No features found for this song.")
            return None
    except Exception as e:
        st.error(f"Error fetching song features: {e}")
        return None

# When a song ID is entered
if song_id:
    song_features = get_song_features(song_id)

    if song_features is not None:
        # Display album cover and track information
        st.image(song_features['Album Cover'].values[0], width=300)
        st.subheader(song_features['Track Name'].values[0])
        st.write(song_features['Artist'].values[0])

        # Drop track name, artist, and album cover columns before processing
        song_features = song_features.drop(columns=['Track Name', 'Artist', 'Album Cover'])

        # Apply the KMeans clustering (after scaling with a_scaler)
        song_features_scaled = a_scaler.transform(song_features)
        song_features['Cluster'] = kmeans.predict(song_features_scaled)

        # Preprocess the Loudness features
        song_features = ewd_lo_disc.transform(song_features)

        # Predict the decade group
        predicted_decade_group = rf.predict(song_features)

        if predicted_decade_group == 0:
            st.markdown("<h3 style='color:#f1c40f;'>📀 This track is giving me the golden nostalgia of the 50s-60s, where it all began!</h3>", unsafe_allow_html=True)
        elif predicted_decade_group == 1:
            st.markdown("<h3 style='color:#e67e22;'>🕺 Whoa! This track really has the essence of the 70s-80s!</h3>", unsafe_allow_html=True)
        elif predicted_decade_group == 2:
            st.markdown("<h3 style='color:#9b59b6;'>🎉 I'm getting some iconic 90s-2000s vibes here!</h3>", unsafe_allow_html=True)
        elif predicted_decade_group == 3:
            st.markdown("<h3 style='color:#e74c3c;'>🔥 This one definitely sounds like a modern banger in the 2010s-2020s!</h3>", unsafe_allow_html=True)

st.text("Last Updated: 12/9/2024 (Updates Yearly)")
st.text("Classification Model Trained on Songs in Spotify All Out '_ _' Playlists")
st.text("By: Hong Kai")