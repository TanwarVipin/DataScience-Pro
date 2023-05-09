import pandas as pd
import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors as NN
from scipy.sparse import csr_matrix as cm
import helper, Preprocess

st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}
</style>
""", unsafe_allow_html=True)

st.image('https://images.projectsgeek.com/2018/07/recommendation.png')
rating = pd.read_csv(r'data/BX-Book-Ratings.csv', sep=';', on_bad_lines='skip',
                     encoding='latin-1')
book = pd.read_csv(r'data/BX-Books.csv', sep=';', on_bad_lines='skip',
                   encoding='latin-1',low_memory=False)
user = pd.read_csv(r'data/BX-Users.csv', sep=';', on_bad_lines='skip',
                   encoding='latin-1')
final_df = Preprocess.preprocess(rating, book, user)
matrix_df = helper.table(final_df)
sparse_df = cm(matrix_df)
title = final_df['Title'].unique().tolist()
title.sort()
select_book = st.selectbox('Select Book ', title)
suggestion = helper.model(sparse_df, select_book, matrix_df)
movie_list = helper.getlist(matrix_df, suggestion)
col1, col2, col3 = st.columns(3,gap='medium')
col4,col5=st.columns(2,gap='medium')
image_url = helper.image_list(book, movie_list)
with col1:
    st.subheader(movie_list[1])
    st.image(image_url[1])
with col2:
    st.subheader(movie_list[2])
    st.image(image_url[2])
with col3:
    st.subheader(movie_list[3])
    st.image(image_url[3])
with col4:
    st.subheader(movie_list[4])
    st.image(image_url[4])
with col5:
    st.subheader(movie_list[5])
    st.image(image_url[5])

