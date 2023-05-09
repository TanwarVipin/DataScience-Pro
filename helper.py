import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN


def table(df):
    df = df.pivot_table(index='Title', columns='User-ID', values='Book-Rating').fillna(0).astype(int)
    return df


def model(df, book, df1):
    nn = NN(algorithm='brute')
    nn.fit(df)
    idx = np.where(df1.index == book)[0][0]
    d, s = nn.kneighbors(df1.iloc[idx, :].values.reshape(1, -1), n_neighbors=6)
    return s


def getlist(df, x):
    for i in x:
        arr = df.iloc[i].index.values
    return arr


def image_list(books, movie_list):
    url = []
    for i in range(len(movie_list)):
        url.append((books[books['Book-Title'] == movie_list[i]])['Image-URL-L'].values[0])
    return url
