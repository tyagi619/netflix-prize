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