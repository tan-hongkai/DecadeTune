# 🎶 DecadeTune
https://decadetune.streamlit.app/

### Overview
DecadeTune is a machine learning project that classifies songs into decade groups by analyzing audio features and referencing popular hits. This project identifies the distinctive sound signatures that characterize music from various eras, from the 50s-60s to the 2010s-2020s.

The model leverages key musical features such as tempo, loudness, acousticness, energy, obtained with the Spotify API

### Problem Statement
DecadeTune aims to enhance algorithmic recommendations by developing a model that identifies a song's release decade group using only its audio features. By capturing each track's unique sound signature and classifying it according to the defining hits of its respective decade group, DecadeTune will possibly enrich the listening experience for users who favor retro or vintage music.

### Business Value Proposition
In the crowded music streaming industry, personalization and user engagement are crucial factors for success. DecadeTune offers a way to go beyond traditional recommendation methods that rely heavily on surface-level metadata (e.g., release dates or artist names). Instead, it provides a deeper understanding of the actual audio characteristics that define each decade.

- More accurate personalized decade-based recommendations, connecting listeners with music that fits their taste.

- Cater to niche preferences such as, offering a richer and more tailored user experience.

- Appeal to users who are enthusiasts of specific music eras.

### What I Learned
1) Deploying machine learning models with a focus on business value, aligning technical outcomes with improvements.

2) The importance of feature engineering when building classifiers to improve accuracy. For example, creating new features from KMeans Clustering.

3) The interpretation of classifier models using metrics like accuracy, precision, recall, and feature importance

4) Using cross validation to to ensure model robustness and stability across different data splits, preventing overfitting and improving generalization to unseen data.

### Notes
- Last Updated: 12/9/2024 (Updates Yearly)
- Classification Model Trained on Songs in Spotify All Out "_ _" Playlists
- By: Hong Kai