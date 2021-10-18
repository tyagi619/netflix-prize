import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Dot

def readInputFile(filename):
    dataframe = []
    with open(filename, "r") as f:
        movieId = None
        while True:
            line = f.readline()
            if not line:
                break

            if line[-1] == '\n':
                line = line[:-1]
            
            if line[-1] == ':':
                movieId = int(line[:-1])
            else:
                userId = int(line.split(',')[0])
                rating = float(line.split(',')[1])
                dataframe.append([userId, movieId, rating])
    dataframe = np.array(dataframe)
    return pd.DataFrame(data=dataframe, columns=['User', 'Movie', 'Rating']).astype({'User':int, 
                                                                                     'Movie':int, 
                                                                                     'Rating':float}
                                                                                    )

def loadData():
    df1 = readInputFile('data/combined_data_1.txt')
    df2 = readInputFile('data/combined_data_2.txt')
    df3 = readInputFile('data/combined_data_3.txt')
    df4 = readInputFile('data/combined_data_4.txt')
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    
    unique_users = list(df['User'].unique())
    unique_movies = list(df['Movie'].unique())
    map_users = {id:i for i,id in enumerate(unique_users)}
    map_movies = {id:i for i,id in enumerate(unique_movies)}
    df['User'] = df['User'].map(map_users)
    df['Movie'] = df['Movie'].map(map_movies)
    num_users = len(unique_users)
    num_movies = len(unique_movies)

    train_data = tf.data.Dataset.from_tensor_slices((df[['User', 'Movie']], df['Rating']))
    train_data = train_data.shuffle(1024, reshuffle_each_iteration=True)
    return train_data, num_users, num_movies

def recommenderModel(num_users, num_movies, latent_dim=20):
    input = Input(shape=(2,))
    x = Embedding(input_dim=num_users,
                  output_dim=latent_dim,
                  input_length=None)(input[:,0])
    y = Embedding(input_dim=num_movies,
                  output_dim=latent_dim,
                  input_length=None)(input[:,1])
    output = Dot(axes=1, normalize=False)([y,x])
    return Model(inputs=input, outputs=output)

def train(train_data, num_users, num_movies, latent_dim=40):
    model = recommenderModel(num_users=num_users, 
                             num_movies=num_movies,
                             latent_dim=latent_dim)
    print(model.summary())

    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]
                 )
    history = model.fit(x=train_data, batch_size=512, epochs=2, verbose=1)

    print(history.history.keys())

def main():
    train_data, num_users, num_movies = loadData()
    train(train_data, num_users, num_movies)

if __name__ == '__main__':
    main()