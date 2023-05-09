import pandas as pd


def preprocess(rating, book, user):
    book = book.rename(columns={'Book-Title': 'Title', 'Book-Author': 'Author', 'Year-Of-Publication': 'Year'})
    x = rating['User-ID'].value_counts() > 200
    index = x[x].index
    rating = rating[rating['User-ID'].isin(index)]
    df = book.merge(rating, on='ISBN', how='inner').drop(['Image-URL-S', 'Image-URL-M','Image-URL-M'], axis=1)
    book_rating = df.groupby('Title')['Book-Rating'].count().reset_index()
    book_rating.rename(columns={'Book-Rating': 'No_of_Rating'}, inplace=True)
    final_df = df.merge(book_rating, on='Title')
    final_df = final_df[final_df['No_of_Rating'] >= 50]
    final_df.drop_duplicates(subset=['Title', 'User-ID'], inplace=True)
    return final_df
