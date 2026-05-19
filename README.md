# Music Genre Classification

Multiclass classification project using Spotify audio features to predict music genres. Built as part of my MSc in Big Data at UNIR, and included in my portfolio as a practical example of neural network implementation with TensorFlow/Keras.

## Problem

Given quantitative audio features from the Spotify API (danceability, energy, loudness, tempo, etc.), predict the genre of a track across 11 categories: Acoustic, Alt-Rock, Blues, Bollywood, Country, HipHop, IndiePop, Instrumental, Metal, Pop and Rock.

## Dataset

Source: [Music Genre Classification on Kaggle](https://www.kaggle.com/datasets/purumalgi/music-genre-classification)

Around 50,000 tracks described by 14 numerical audio features and labelled with one of 11 genres. The dataset is included in this repository so the notebook can load it directly without authentication.

## What's in the notebook

- Exploratory data analysis: class distribution, correlation heatmap, feature distributions by genre
- Preprocessing: StandardScaler normalization, missing value handling
- Three models trained and compared on the same 80/20 stratified split:
  - Neural network (Keras/TensorFlow 2): 3 hidden layers, ReLU activations, Softmax output, EarlyStopping
  - Random Forest (scikit-learn): 200 estimators
  - Logistic Regression (scikit-learn): multinomial
- Evaluation: accuracy, F1-score macro, confusion matrix and per-class report
- Results comparison table and chart

## Results

| Model | Accuracy | F1-Score (macro) |
|---|---|---|
| Neural Network | — | — |
| Random Forest | — | — |
| Logistic Regression | — | — |

*To be updated after running the notebook.*

## How to run

```bash
git clone https://github.com/Finarosalina/music-genre-classification.git
cd music-genre-classification
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
jupyter notebook Act2_Clasificacion_MusicGenre.ipynb
```

## Stack

Python, TensorFlow/Keras, scikit-learn, Pandas, Matplotlib, Seaborn

## Author

María Pais Fajín — [LinkedIn](https://www.linkedin.com/in/mariapais) · [GitHub](https://github.com/Finarosalina)
