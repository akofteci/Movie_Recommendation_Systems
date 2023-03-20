import pandas as pd
import numpy as np
import streamlit as st
import datetime
import pickle
from joblib import dump,load
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
import re

credits=pd.read_csv('tmdb_5000_credits.csv')
movies=pd.read_csv('tmdb_5000_movies.csv')
total_movies=pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')




movies = movies.merge(credits, on='title')
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['spoken_languages'] = movies['spoken_languages'].apply(convert)
movies['production_companies'] = movies['production_companies'].apply(convert)
movies['production_countries'] = movies['production_countries'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies['crew'] = movies['crew'].apply(convert)

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew','vote_average','vote_count','popularity']]





## BEST SCORED
C = movies["vote_average"].mean()
m=movies["vote_count"].quantile(0.85)

q_movies = movies.copy().loc[movies['vote_count'] >= m]

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies[['title', 'vote_count', 'vote_average', 'score','genres']].head(10)

a_movies = q_movies.copy()
a_movies=a_movies[['title', 'vote_count', 'vote_average', 'score','genres']]
def gender_selector(a):
    return a_movies[a_movies['genres'].str.contains(a, regex=False)]





## CONTENT BASE

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title','genres']].iloc[movie_indices]




##COLLOBRATIVE FILTERING

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]","",title)

total_movies['title']=total_movies['title'].apply(clean_title)


def rating_cor_movies(title):
    movie_id = int(total_movies[total_movies['title'] == title]['movieId'].values)
    sim_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] >= 4)]["userId"].unique()
    sim_user_recs = ratings[(ratings["userId"].isin(sim_users) & (ratings["rating"] >= 4))]["movieId"]

    sim_user_recs = sim_user_recs.value_counts(()) / len(sim_users)
    sim_user_recs = sim_user_recs[sim_user_recs > 0.1]

    all_users = ratings[(ratings["movieId"].isin(sim_user_recs.index)) & (ratings["rating"] > 4)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([sim_user_recs, all_users_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    return rec_percentages.head(10).merge(total_movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]









##STREAMLIT
st.set_page_config(page_title="MRS")
st.sidebar.image('IMDB_Logo_.jpg')
st.header("Here are the best Scored Movies According to IMDB users")

genre = st.radio("What\'s your favorite movie genre",('Comedy', 'Drama', 'Action','Thriller', 'Science Fiction', 'Mystery'),horizontal=True)

st.write(gender_selector(genre))
st.sidebar.write("You will have 100.000 movie choices")

st.title("Content Base Movie Recommendation System")
st.markdown("Aydın ")
st.sidebar.image("unnamed.jpg")

movielist = movies['title'].values

liked=st.selectbox("**Beğendiğiniz Film**", movielist)

if st.button('Recommend'):
    st.success('Here are our Recomendations for you', icon="✅")
    st.write(get_recommendations(liked, cosine_sim=cosine_sim))

st.title("Collabrative Movie Recommendation System")
total_movielist = total_movies['title'].values
mliked=st.selectbox("**Beğendiğiniz Film2**", total_movielist)
if st.button('Recommendd'):
    st.success('Here are our Recomendations for you', icon="✅")
    st.write(rating_cor_movies(mliked))

st.write('     ')
st.write('   ')
st.write('   ')
st.write('   ')
st.write('   ')
st.write('   ')
st.write('   ')
st.write('   ')
st.write('   ')
st.write('   ')
st.write('   ')
st.write('   ')
st.write('   ')
st.write('   ')

renk = ['Green', 'Yellow', 'Red', 'Blue']
options = st.multiselect(
    'What are your favorite colors', renk
    ,
    ['Yellow', 'Red'])

st.write('You favorite movies:', options)

t = st.time_input('Set an alarm for', datetime.time(8, 45))
st.write('Alarm is set for', t)



if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')