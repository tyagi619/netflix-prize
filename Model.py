import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, InputLayer, InputSpec

from Layers import SVD, Bias

class BayesianModel():
    def __init__(self, users, movies, K=25):
        self.movie_bias = None
        self.user_bias = None
        self.num_movies = movies
        self.num_users = users
        self.K = K

    def train(self, ratings_df, save_to):
        user_bias, movie_bias = bayesianAverage(ratings_df, self.num_users, self.num_movies, self.K)
        self.user_bias = user_bias['Avg Rating Offset'].values
        self.movie_bias = movie_bias['Avg Rating'].values
        np.savetxt(save_to + '/movie_bias.csv', self.movie_bias, delimiter=',', fmt='%.2f')
        np.savetxt(save_to + '/user_bias.csv', self.user_bias, delimiter=',', fmt='%.2f')

    def test(self, X_test, y_test):
        if self.user_bias is None or self.movie_bias is None:
            logging.error("No model weights. Either train model to call load_weights to load pretrained weights")
            raise RuntimeError("No model weights found")

        y_pred = self.movie_bias[X_test['movies']] + self.user_bias[X_test['users']]
        rmse = np.sqrt(np.mean((y_pred-y_test)**2))
        return rmse, y_pred
    
    def recommend(self, user_id):
        if self.user_bias is None or self.movie_bias is None:
            logging.error("No model weights. Either train model to call load_weights to load pretrained weights")
            raise RuntimeError("No model weights found")

        y_pred = self.movie_bias + self.user_bias[user_id]
        return y_pred

    def load_weights(self, filepath):
        try:
            self.movie_bias = np.loadtxt(filepath + '/movie_bias.csv', delimiter=',')
            self.user_bias = np.loadtxt(filepath + '/user_bias.csv', delimiter=',')
        except:
            logging.error("Unable to read model files")
            raise RuntimeError("Unable to read model files")


class SimpleSVD(Model):
    def __init__(self, users, movies, latent_dim, use_sigmoid=False, **kwargs):
        super(SimpleSVD, self).__init__(**kwargs)
        self._users = users
        self._movies = movies
        self._latent_dim = latent_dim
        self._use_sigmoid = use_sigmoid
        self._matrix = None
        self.input_spec = {'users': InputSpec(shape=(None,), allow_last_axis_squeeze=True),
                           'movies': InputSpec(shape=(None,), allow_last_axis_squeeze=True)
                          }
        self.build()
    
    def build(self, input_shape={'users':(None,), 'movies':(None,)}):
        self._input_users = InputLayer(input_shape=(), batch_size=None, name='users')
        self._input_movies = InputLayer(input_shape=(), batch_size=None, name='movies')
        self._matrix = SVD(users=self._users,
                           movies=self._movies,
                           latent_dim=self._latent_dim,
                           name='rating_matrix'
                          )
        input = {'users':Input(()), 'movies':Input(())}
        output = self.call(input)
        super(SimpleSVD, self).build(input_shape)

    def call(self, inputs, training=False):
        u = inputs['users']
        m = inputs['movies']
        if u.shape.rank == 2:
            u = tf.squeeze(u, axis=-1)
        if m.shape.rank == 2:
            m = tf.squeeze(m, axis=-1)
        r = self._matrix(u, m)
        if self._use_sigmoid:
            r = tf.keras.activations.sigmoid(r)
            r = tf.math.scalar_mul(5.0, r)
        return r

    def recommend(self, user_id, batch_size=2048):
        movies_list = np.array([i for i in range(self._movies)])
        user_list = np.array([user_id for _ in range(self._movies)])
        X_test = {'users': user_list, 'movies':movies_list}
        y_pred = self.predict(x=X_test, batch_size=batch_size)
        return y_pred[:,0]


class SVDImproved(Model):
    def __init__(self, users, movies, latent_dim, K=25, data=None, use_sigmoid=False, **kwargs):
        super(SVDImproved, self).__init__(**kwargs)
        self._users = users
        self._movies = movies
        self._latent_dim = latent_dim
        self._K = K
        self._use_sigmoid = use_sigmoid
        self._matrix = None
        self._bias = None
        self.input_spec = {'users': InputSpec(shape=(None,), allow_last_axis_squeeze=True),
                           'movies': InputSpec(shape=(None,), allow_last_axis_squeeze=True)
                          }
        self.build(data=data)
    
    def build(self, input_shape={'users':(None,), 'movies':(None,)}, data=None):
        self._input_users = InputLayer(input_shape=(), batch_size=None, name='users')
        self._input_movies = InputLayer(input_shape=(), batch_size=None, name='movies')
        self._matrix = SVD(users=self._users,
                           movies=self._movies,
                           latent_dim=self._latent_dim,
                           name='rating_matrix'
                          )
        if data is None:
            user_bias = np.zeros(shape=(self._users,), dtype=float)
            movie_bias = np.zeros(shape=(self._movies,), dtype=float)
        else:
            user_bias, movie_bias = bayesianAverage(data, self._users, self._movies, self._K)
            user_bias = user_bias['Avg Rating Offset'].values
            movie_bias = movie_bias['Avg Rating']
        self._bias = Bias(users=self._users,
                          movies=self._movies, 
                          trainable_bias=False, 
                          user_bias=user_bias, 
                          movie_bias=movie_bias,
                          name='biases'
                         )
        input = {'users':Input(()), 'movies':Input(())}
        output = self.call(input)
        super(SVDImproved, self).build(input_shape)
    
    def call(self, inputs, training=False):
        u = inputs['users']
        m = inputs['movies']
        if u.shape.rank == 2:
            u = tf.squeeze(u, axis=-1)
        if m.shape.rank == 2:
            m = tf.squeeze(m, axis=-1)
        r = self._matrix(u, m)
        b = self._bias(u, m)
        if self._use_sigmoid:
            r = tf.keras.activations.sigmoid(r)
            r = tf.math.scalar_mul(5.0, r)
        r = tf.keras.layers.add([r,b])
        if self._use_sigmoid:
            r = tf.math.scalar_mul(0.5, r)
        return r
    
    def recommend(self, user_id, batch_size=2048):
        movies_list = np.array([i for i in range(self._movies)])
        user_list = np.array([user_id for _ in range(self._movies)])
        X_test = {'users': user_list, 'movies':movies_list}
        y_pred = self.predict(x=X_test, batch_size=batch_size)
        return y_pred[:,0]

def bayesianAverage(data, num_users, num_movies, K):
    # Calculate avg rating for each movie
    global_movie_avg = data['Rating'].mean()
    count_movie_ratings = data.groupby('Movie').size().reset_index(drop=False).rename(columns={0:'Num Ratings'})
    sum_movie_ratings = data.groupby('Movie')['Rating'].sum().reset_index(drop=False).rename(columns={'Rating':'Sum Ratings'})
    movie_avg_df = sum_movie_ratings.merge(count_movie_ratings, on='Movie', how='inner')
    movie_avg_df['Avg Rating'] = (movie_avg_df['Sum Ratings'] + K * global_movie_avg)/(K + movie_avg_df['Num Ratings'])

    movie_bias = pd.DataFrame({'Movie': range(num_movies)})
    movie_bias = movie_bias.merge(movie_avg_df[['Movie', 'Avg Rating']], on='Movie', how='left')
    movie_bias.sort_values(by='Movie', inplace=True)
    assert(movie_bias['Avg Rating'].notna().any())

    # Get user ratings offset
    data = data.merge(movie_bias, on='Movie', how='left')
    assert(data['Avg Rating'].notna().any())
    data['Rating Offset'] = data['Rating'] - data['Avg Rating']
    
    # Calculate avg rating offset for each user
    global_user_offset_avg = data['Rating Offset'].mean()
    count_user_ratings = data.groupby('User').size().reset_index(drop=False).rename(columns={0:'Num Ratings'})
    sum_user_ratings_offset = data.groupby('User')['Rating Offset'].sum().reset_index(drop=False).rename(columns={'Rating Offset':'Sum Ratings Offset'})
    user_avg_offset_df = sum_user_ratings_offset.merge(count_user_ratings, on='User', how='inner')
    user_avg_offset_df['Avg Rating Offset'] = (user_avg_offset_df['Sum Ratings Offset'] + K * global_user_offset_avg)/(K + user_avg_offset_df['Num Ratings'])

    user_bias = pd.DataFrame({'User':range(num_users)})
    user_bias = user_bias.merge(user_avg_offset_df[['User', 'Avg Rating Offset']], on='User', how='left')
    user_bias.sort_values(by='User', inplace=True)
    assert(user_bias['Avg Rating Offset'].notna().any())

    return user_bias, movie_bias