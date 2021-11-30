import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, InputLayer, InputSpec

from Layers import SVD, Bias

class Recommender(Model):
    def __init__(self, users, movies, latent_dim, use_bias=False, global_bias=False, **kwargs):
        super(Recommender, self).__init__(**kwargs)
        self._users = users
        self._movies = movies
        self._latent_dim = latent_dim
        self._use_bias = use_bias
        self._global_bias = (use_bias and global_bias)
        self._matrix = None
        self._bias = None
        self.input_spec = {
                           'users':InputSpec(shape=(None,), allow_last_axis_squeeze=True),
                           'movies':InputSpec(shape=(None,), allow_last_axis_squeeze=True)
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
        if self._use_bias:
            self._bias = Bias(users=self._users,
                              movies=self._movies,
                              global_bias=self._global_bias,
                              name='biases'
                             )
        input = {'users':Input(()), 'movies':Input(())}
        output = self.call(input)
        super(Recommender, self).build(input_shape)
    
    def call(self, inputs, training=False):
        u = inputs['users']
        m = inputs['movies']
        if u.shape.rank == 2:
            u = tf.squeeze(u, axis=-1)
        if m.shape.rank == 2:
            m = tf.squeeze(m, axis=-1)
        r = self._matrix(u, m)
        if self._use_bias:
            b = self._bias(u, m)
            r = tf.keras.layers.add([r,b])
        r = tf.keras.activations.sigmoid(r)
        return tf.math.scalar_mul(5.0, r)

class BayesianModel():
    def __init__(self, users, movies, K=25):
        self.bi = None
        self.bj = None
        self.num_movies = movies
        self.num_users = users
        self.K = 25

    def train(self, ratings_df, save_to):
        movie_biases = pd.DataFrame({'Movie': range(self.num_movies)})
        global_average_movies = ratings_df['Rating'].mean()

        avg_movie_ratings = ratings_df.groupby('Movie')['Rating'].sum().reset_index(drop=False).rename(columns={'Rating':'sum'})
        num_movie_ratings = ratings_df.groupby('Movie')['User'].size().reset_index(drop=False).rename(columns={'User':'count'})
        movie_biases = movie_biases.merge(avg_movie_ratings, how='left', on='Movie')
        movie_biases = movie_biases.merge(num_movie_ratings, how='left', on='Movie')
        movie_biases['bi'] = (movie_biases['sum'] + self.K * global_average_movies)/(movie_biases['count'] + self.K)
        movie_biases = movie_biases[['Movie', 'bi']].fillna(0).sort_values(by='Movie')
        self.bi = movie_biases['bi'].values

        user_biases = pd.DataFrame({'User': range(self.num_users)})
        user_offset_df = ratings_df.merge(movie_biases, how='left', on='Movie')[['User', 'bi', 'Rating']]
        user_offset_df['offset'] = user_offset_df['Rating'] - user_offset_df['bi']
        user_offset_df = user_offset_df[['User', 'offset']]
        global_average_user_offset = user_offset_df['offset'].mean()

        avg_user_biases = user_offset_df.groupby('User')['offset'].sum().reset_index(drop=False).rename(columns={'offset':'sum'})
        num_user_ratings = user_offset_df.groupby('User')['offset'].size().reset_index(drop=False).rename(columns={'offset':'count'})
        user_biases = user_biases.merge(avg_user_biases, how='left', on='User')
        user_biases = user_biases.merge(num_user_ratings, how='left', on='User')
        user_biases['bj'] = (user_biases['sum'] + self.K * global_average_user_offset)/(user_biases['count'] + self.K)
        user_biases = user_biases[['User', 'bj']].fillna(0).sort_values(by='User')
        self.bj = user_biases['bj'].values

        np.savetxt(save_to + '/bi.csv', self.bi, delimiter=',', fmt='%.2f')
        np.savetxt(save_to + '/bj.csv', self.bj, delimiter=',', fmt='%.2f')

    def test(self, X_test, y_test):
        if self.bi is None or self.bj is None:
            logging.error("No model weights. Either train model to call load_weights to load pretrained weights")
            raise RuntimeError("No model weights found")

        y_pred = self.bi[X_test['movies']] + self.bj[X_test['users']]
        rmse = np.sqrt(np.mean((y_pred-y_test)**2))
        return rmse, y_pred
    
    def predict(self, user_id):
        if self.bi is None or self.bj is None:
            logging.error("No model weights. Either train model to call load_weights to load pretrained weights")
            raise RuntimeError("No model weights found")

        movies_list = np.array([i for i in range(self.num_movies)])
        user_list = np.array([user_id for _ in range(self.num_movies)])
        X_test = {'users': user_list, 'movies':movies_list}
        y_pred = self.bi[X_test['movies']] + self.bj[X_test['users']]
        return y_pred

    def load_weights(self, filepath):
        try:
            self.bi = np.loadtxt(filepath + '/bi.csv', delimiter=',')
            self.bj = np.loadtxt(filepath + '/bj.csv', delimiter=',')
        except:
            logging.error("Unable to read model files")
            raise RuntimeError("Unable to read model files")