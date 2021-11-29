"""
Usage:
    run.py train baseline --probe-src=<file> --train-src=<file> [options]
    run.py recommend baseline [options] MODEL_PATH USER_MAP MOVIE_MAP USER_ID OUTPUT_FILE
    run.py test baseline [options] MODEL_PATH USER_MAP MOVIE_MAP TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --train-src=<file>                      train source file
    --probe-src=<file>                      validation source file
    --movie-name-file=<file>                csv mapping movie id to movie name[default: ./data/movie_titles.csv]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 2048]
    --latent-dim=<int>                      latent dimension [default: 40]
    --max-epoch=<int>                       max epoch [default: 10]
    --patience=<int>                        wait for how many iterations to end training [default: 2]
    --lr=<float>                            learning rate [default: 0.001]
    --use-bias=<int>                        include user and movie bias paramters in learning [default: 0]
    --use-global-bias=<int>                 include global bias parameter in learning [default: 0]
    --save-to=<file>                        model save path [default: ./output/model]
    --save-xval-to=<file>                   src validation data save path [default: ./output/x_test.csv]
    --save-yval-to=<file>                   tgt validation data save path [default: ./output/y_test.csv]
    --save-user-map-to=<file>               user mapping save path [default: ./output/user.json]
    --save-movie-map-to=<file>              movie mapping save path [default: ./output/movie.json]
    --map-input=<int>                       specifies whether to map input using mapping file [default: 1]
"""

from datetime import datetime
import os
import sys
import json
import logging
from docopt import docopt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from Model import BayesianModel, Recommender

def readTrainFile(filename):
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
                                                                                     'Rating':float
                                                                                    })

def readProbeFile(filename):
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
                dataframe.append([userId, movieId])
    dataframe = np.array(dataframe)
    return pd.DataFrame(data=dataframe, columns=['User', 'Movie']).astype({'User':int, 
                                                                           'Movie':int,
                                                                         })

def loadData(inputFiles, probe_file):

    logging.info('reading probe data...')
    probe_data = readProbeFile(probe_file)
    logging.info('reading probe data complete')
    
    probe_list = []
    df_list = []
    
    logging.info('reading input file...')
    for i,file in enumerate(inputFiles):
        logging.info('reading file: %d/%d', i+1, len(inputFiles))
        train_df = readTrainFile(file)
        train_set = train_df.merge(probe_data, how='outer', left_on=['User', 'Movie'], right_on=['User', 'Movie'], indicator=True)
        train_df = train_set.loc[lambda x: x['_merge']=='left_only'][['User', 'Movie', 'Rating']].reset_index(drop=True)
        probe_df = train_set.loc[lambda x: x['_merge']=='both'][['User', 'Movie', 'Rating']].reset_index(drop=True)
        df_list.append(train_df)
        probe_list.append(probe_df)
    logging.info('reading input files complete')
    
    train_df = pd.concat(df_list, ignore_index=True)
    probe_df = pd.concat(probe_list, ignore_index=True)
    logging.info('probe dataset contains %d rows', probe_df.shape[0])
    logging.info('train dataset contains %d rows', train_df.shape[0])
    logging.info('rows in training set: %d', train_df.shape[0] + probe_df.shape[0])
    print('='*100)
    return train_df, probe_df

def preprocessData(train_data, probe_data, args):

    logging.info('writing probe data to disk...')
    np.savetxt(args['--save-xval-to'], probe_data[['User', 'Movie']].values, fmt='%d', delimiter=',', newline='\n')
    np.savetxt(args['--save-yval-to'], probe_data[['Rating']].values, fmt='%.1f', delimiter=',', newline='\n')
    logging.info('write probe data to %s and %s complete', args['--save-xval-to'], args['--save-yval-to'])
    print('='*100)

    logging.info('mapping users to continuous series...')
    unique_users_probe = set(probe_data['User'].unique().tolist())
    unique_users_train = set(train_data['User'].unique().tolist())
    unique_users = list(unique_users_train.union(unique_users_probe))
    num_users = len(unique_users)
    user_mapping = {id:i for i, id in enumerate(unique_users)}
    train_data['User'] = train_data['User'].map(user_mapping)
    probe_data['User'] = probe_data['User'].map(user_mapping)
    logging.info('mapping users complete')

    logging.info('saving user mapping to disk...')
    userMappingFile = args['--save-user-map-to']
    with open(userMappingFile, "w") as f:
        json.dump(user_mapping, f)
    logging.info('saved user mapping to %s', args['--save-user-map-to'])
    print('='*100)

    logging.info('mapping movies to continuous series...')
    unique_movies_probe = set(probe_data['Movie'].unique().tolist())
    unique_movies_train = set(train_data['Movie'].unique().tolist())
    unique_movies = list(unique_movies_train.union(unique_movies_probe))
    num_movies = len(unique_movies)
    movie_mapping = {id:i for i, id in enumerate(unique_movies)}
    train_data['Movie'] = train_data['Movie'].map(movie_mapping)
    probe_data['Movie'] = probe_data['Movie'].map(movie_mapping)
    logging.info('mapping movies complete')

    logging.info('saving movie mapping to disk...')
    movieMappingFile = args['--save-movie-map-to']
    with open(movieMappingFile, "w") as f:
        json.dump(movie_mapping, f)
    logging.info('saved movie mapping to %s', args['--save-movie-map-to'])
    print('='*100)

    return train_data, probe_data, num_users, num_movies

def getTrainTestData(train_data, probe_data, batch_size, args):
    
    X_train = train_data[['User', 'Movie']].values
    y_train = train_data[['Rating']].values
    X_test  = probe_data[['User', 'Movie']].values
    y_test  = probe_data[['Rating']].values 
    logging.info('X_train contains %d rows', X_train.shape[0])
    logging.info('X_val contains %d rows', X_test.shape[0])

    logging.info('converting train data to tensorflow Dataset....')
    x_train_user = tf.convert_to_tensor(X_train[:,0])
    x_train_movie = tf.convert_to_tensor(X_train[:,1])
    y_train = tf.convert_to_tensor(y_train)
    train_data = tf.data.Dataset.from_tensor_slices(({'users':x_train_user, 'movies':x_train_movie}, y_train))
    train_data = train_data.shuffle(2*batch_size, reshuffle_each_iteration=True)
    train_data = train_data.batch(batch_size)
    logging.info('train data converted to tensorflow Dataset')

    logging.info('converting validation data to tensorflow Dataset...')
    x_val_user = tf.convert_to_tensor(X_test[:,0])
    x_val_movie = tf.convert_to_tensor(X_test[:,1])
    y_val = tf.convert_to_tensor(y_test)
    val_data = tf.data.Dataset.from_tensor_slices(({'users':x_val_user, 'movies':x_val_movie}, y_val))
    val_data = val_data.batch(batch_size)
    logging.info('validation data converted to tensorflow Dataset')
    print('='*100)
    
    return train_data, val_data

def trainBaseline(args):

    train_files = args['--train-src'].split(',')
    probe_file  = args['--probe-src']
    model_save_path = args['--save-to']
    
    if len(train_files)==0:
        logging.error('empty input file list passed')
        raise RuntimeError('No input files for training.Use --train-src argument to pass input files')

    train_data, probe_data = loadData(train_files, probe_file)
    train_data, probe_data, num_users, num_movies = preprocessData(train_data, probe_data, args)
    logging.info('num users: %d', num_users)
    logging.info('num movies: %d', num_movies)

    model = BayesianModel(users=num_users,
                          movies=num_movies,
                          K=25
                         )

    model.train(ratings_df=train_data, save_to=model_save_path)

def testBaseline(args):
    
    model_path = args['MODEL_PATH']
    user_map_file = args['USER_MAP']
    movie_map_file = args['MOVIE_MAP']
    x_test_file = args['TEST_SOURCE_FILE']
    y_test_file = args['TEST_TARGET_FILE']

    map_input = bool(int(args['--map-input']))
    logging.info('map input: %d', map_input)
    print('='*100)

    try:
        X_test = np.loadtxt(x_test_file, delimiter=',', dtype=int)
        logging.info('loaded input file %s', x_test_file)
    except:
        logging.error('unable to read %s', x_test_file)
        raise RuntimeError('Unable to read Input File')

    try:
        y_test = np.loadtxt(y_test_file, delimiter=',', dtype=float)
        logging.info('loaded input file %s', y_test_file)
    except:
        logging.error('unable to read %s', y_test_file)
        raise RuntimeError('Unable to read Target Input File')

    if user_map_file:
        try:
            with open(user_map_file, 'r') as f:
                user_map = json.loads(f.read())
                user_map = {int(k):int(v) for k,v in user_map.items()}
            logging.info('read %s success', user_map_file)
        except Exception as e:
            logging.info('%s', e)
            user_map_file = None
            logging.warning('unable to open user map file %s. setting user map file to None', user_map_file)

    if movie_map_file:
        try:
            with open(movie_map_file, 'r') as f:
                movie_map = json.loads(f.read())
                movie_map = {int(k):int(v) for k,v in movie_map.items()}
            logging.info('read %s success', movie_map_file)
        except Exception as e:
            logging.info('%s', e)
            movie_map_file = None
            logging.warning('unable to open movie map file %s. setting movie map file to None', movie_map_file)

    if map_input:
        if user_map_file:
            X_test[:,0] = np.array([user_map[i] for i in X_test[:,0].tolist()])
            logging.info('map users using mapping file')
            num_users = len(user_map)
        else:
            logging.error('no mapping file found for user.')
            raise RuntimeError('No mapping file found for user to map input.')
        
        if movie_map_file:
            X_test[:,1] = np.array([movie_map[i] for i in X_test[:,1].tolist()])
            logging.info('map movies using mapping file')
            num_movies = len(movie_map)
        else:
            logging.error('no mapping file found for user.')
            raise RuntimeError('No mapping file found for movies to map input.')
    else:
        if user_map_file:
            num_users = len(user_map)
        else:
            logging.warning('no mapping file specified for user. using max of input file to get num users')
            num_users = np.max(X_test[:,0]).item()

        if movie_map_file:
            num_movies = len(movie_map)
        else:
            logging.warning('no mapping file specified for movie. using max of input file to get num movies')
            num_movies = np.max(X_test[:,1]).item()

    logging.info('num users : %d', num_users)    
    logging.info('num movies : %d', num_movies)
    X_test = {'users': X_test[:,0], 'movies':X_test[:,1]}
    print('='*100)

    model = BayesianModel(users=num_users, 
                          movies=num_movies,
                          K=25
                         )
    model.load_weights(filepath=model_path)
    logging.info('successfully loaded trained model weights')

    rmse, y_pred = model.test(X_test, y_test)

    logging.info('RMSE : %0.4f', rmse)

    correct_pred = np.sum(np.where(abs(y_pred - y_test) < 0.5, 1, 0))
    logging.info('num correct predictions: %d/%d', correct_pred, y_test.shape[0])
    var_pred = np.sum(np.square(y_pred - y_test)) / y_test.shape[0]
    logging.info('variance in pred data : %0.3f', var_pred)
    
    np.savetxt(args['OUTPUT_FILE'], y_pred, fmt='%.1f', delimiter=',', newline='\n')
    logging.info('saved output file to %s', args['OUTPUT_FILE'])


def trainSVD(args):

    train_files = args['--train-src'].split(',')
    probe_file  = args['--probe-src']
    
    if len(train_files)==0:
        logging.error('empty input file list passed')
        raise RuntimeError('No input files for training.Use --train-src argument to pass input files')

    train_batch_size = int(args['--batch-size'])
    logging.info('batch size: %d', train_batch_size)
    latent_dim = int(args['--latent-dim'])
    logging.info('latent dim: %d', latent_dim)
    max_epochs = int(args['--max-epoch'])
    logging.info('max epochs: %d', max_epochs)
    patience = int(args['--patience'])
    logging.info('patience: %d', patience)
    lr = float(args['--lr'])
    logging.info('learning rate: %.4f', lr)
    use_bias = bool(int(args['--use-bias']))
    logging.info('use bias: %d', use_bias)
    use_global_bias = bool(int(args['--use-global-bias']))
    logging.info('use global bias: %d', use_global_bias)
    model_save_path = args['--save-to']
    print('='*100)

    train_data, probe_data = loadData(train_files, probe_file)
    train_data, probe_data, num_users, num_movies = preprocessData(train_data, probe_data, args)
    logging.info('num users: %d', num_users)
    logging.info('num movies: %d', num_movies)
    train_data, val_data = getTrainTestData(train_data, probe_data, train_batch_size, args)

    model = Recommender(users=num_users, 
                        movies=num_movies,
                        latent_dim=latent_dim,
                        use_bias=use_bias,
                        global_bias=use_global_bias)
    print(model.summary())
    print('='*100)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]
                 )

    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                         monitor='val_root_mean_squared_error',
                                                         save_best_only=True,
                                                         mode='min',
                                                         save_freq='epoch',
                                                         save_weights_only=True,
                                                         verbose=1
                                                        )
    
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                                     mode='min',
                                                     min_delta=0.01,
                                                     patience=patience,
                                                     verbose=1)
    
    history = model.fit(x=train_data, 
                        validation_data=val_data, 
                        epochs=max_epochs, 
                        callbacks=[modelCheckpoint, earlyStopping],
                        verbose=1
                       )

def test(args):
    
    model_path = args['MODEL_PATH']
    user_map_file = args['USER_MAP']
    movie_map_file = args['MOVIE_MAP']
    x_test_file = args['TEST_SOURCE_FILE']
    y_test_file = args['TEST_TARGET_FILE']

    batch_size = int(args['--batch-size'])
    logging.info('batch size: %d', batch_size)
    latent_dim = int(args['--latent-dim'])
    logging.info('latent dim: %d', latent_dim)
    use_bias = bool(int(args['--use-bias']))
    logging.info('use bias: %d', use_bias)
    use_global_bias = bool(int(args['--use-global-bias']))
    logging.info('use global bias: %d', use_global_bias)
    map_input = bool(int(args['--map-input']))
    logging.info('map input: %d', map_input)
    print('='*100)

    try:
        X_test = np.loadtxt(x_test_file, delimiter=',', dtype=int)
        logging.info('loaded input file %s', x_test_file)
    except:
        logging.error('unable to read %s', x_test_file)
        raise RuntimeError('Unable to read Input File')

    try:
        y_test = np.loadtxt(y_test_file, delimiter=',', dtype=float)
        logging.info('loaded input file %s', y_test_file)
    except:
        logging.error('unable to read %s', y_test_file)
        raise RuntimeError('Unable to read Target Input File')

    if user_map_file:
        try:
            with open(user_map_file, 'r') as f:
                user_map = json.loads(f.read())
                user_map = {int(k):int(v) for k,v in user_map.items()}
            logging.info('read %s success', user_map_file)
        except Exception as e:
            logging.info('%s', e)
            user_map_file = None
            logging.warning('unable to open user map file %s. setting user map file to None', user_map_file)

    if movie_map_file:
        try:
            with open(movie_map_file, 'r') as f:
                movie_map = json.loads(f.read())
                movie_map = {int(k):int(v) for k,v in movie_map.items()}
            logging.info('read %s success', movie_map_file)
        except Exception as e:
            logging.info('%s', e)
            movie_map_file = None
            logging.warning('unable to open movie map file %s. setting movie map file to None', movie_map_file)

    if map_input:
        if user_map_file:
            X_test[:,0] = np.array([user_map[i] for i in X_test[:,0].tolist()])
            logging.info('map users using mapping file')
            num_users = len(user_map)
        else:
            logging.error('no mapping file found for user.')
            raise RuntimeError('No mapping file found for user to map input.')
        
        if movie_map_file:
            X_test[:,1] = np.array([movie_map[i] for i in X_test[:,1].tolist()])
            logging.info('map movies using mapping file')
            num_movies = len(movie_map)
        else:
            logging.error('no mapping file found for user.')
            raise RuntimeError('No mapping file found for movies to map input.')
    else:
        if user_map_file:
            num_users = len(user_map)
        else:
            logging.warning('no mapping file specified for user. using max of input file to get num users')
            num_users = np.max(X_test[:,0]).item()

        if movie_map_file:
            num_movies = len(movie_map)
        else:
            logging.warning('no mapping file specified for movie. using max of input file to get num movies')
            num_movies = np.max(X_test[:,1]).item()

    logging.info('num users : %d', num_users)    
    logging.info('num movies : %d', num_movies)
    X_test = {'users': X_test[:,0], 'movies':X_test[:,1]}
    print('='*100)

    model = Recommender(users=num_users, 
                        movies=num_movies,
                        latent_dim=latent_dim,
                        use_bias=use_bias,
                        global_bias=use_global_bias)
    model.load_weights(filepath=model_path).expect_partial()
    logging.info('successfully loaded trained model weights')

    model.compile(loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()]
                 )

    loss = model.evaluate(x=X_test, y=y_test, batch_size=batch_size)
    y_pred = model.predict(x=X_test, batch_size=batch_size)

    if type(loss) == type([]):
        for metrics, value in zip(model.metrics_names, loss):
            logging.info('%s : %0.4f', metrics, value)
    else:
        logging.info('%s : %0.4f', model.metrics_names[0], loss)

    correct_pred = np.sum(np.where(abs(y_pred[:,0] - y_test) < 0.5, 1, 0))
    logging.info('num correct predictions: %d/%d', correct_pred, y_test.shape[0])
    var_pred = np.sum(np.square(y_pred[:,0] - y_test)) / y_test.shape[0]
    logging.info('variance in pred data : %0.3f', var_pred)
    
    np.savetxt(args['OUTPUT_FILE'], y_pred, fmt='%.1f', delimiter=',', newline='\n')
    logging.info('saved output file to %s', args['OUTPUT_FILE'])

def recommend(args):
    
    model_path = args['MODEL_PATH']
    user_map_file = args['USER_MAP']
    movie_map_file = args['MOVIE_MAP']
    movie_names_file = args['--movie-name-file']

    batch_size = int(args['--batch-size'])
    logging.info('batch size: %d', batch_size)
    latent_dim = int(args['--latent-dim'])
    logging.info('latent dim: %d', latent_dim)
    use_bias = bool(int(args['--use-bias']))
    logging.info('use bias: %d', use_bias)
    use_global_bias = bool(int(args['--use-global-bias']))
    logging.info('use global bias: %d', use_global_bias)
    map_input = bool(int(args['--map-input']))
    logging.info('map input: %d', map_input)
    print('='*100)

    try:
        with open(user_map_file, 'r') as f:
            user_map = json.loads(f.read())
            user_map = {int(k):int(v) for k,v in user_map.items()}
        logging.info('read %s success', user_map_file)
    except Exception as e:
        logging.info('%s', e)
        logging.error('unable to open user map file %s', user_map_file)
        raise RuntimeError('User mapping file %s not found', user_map_file)

    try:
        with open(movie_map_file, 'r') as f:
            movie_map = json.loads(f.read())
            movie_map = {int(k):int(v) for k,v in movie_map.items()}
        logging.info('read %s success', movie_map_file)
    except Exception as e:
        logging.info('%s', e)
        logging.error('unable to open movie map file %s', movie_map_file)
        raise RuntimeError('Movie mapping file %s not found', movie_map_file)

    num_users = len(user_map)
    logging.info('num users: %d', num_users)
    num_movies = len(movie_map)
    logging.info('num movies: %d', num_movies)

    try:
        movie_names = pd.read_csv(movie_names_file, names=['movie_id', 'year', 'name'])
    except:
        logging.error('unable to read movie names file %s', )
        raise RuntimeError('Unable to read Input File')
    
    movie_names['movie_id'] = movie_names['movie_id'].map(movie_map)

    if map_input:
        user_id = user_map[int(args['USER_ID'])]
    else:
        user_id = int(args['USER_ID'])
    
    movies_list = np.array([i for i in range(num_movies)])
    user_list = np.array([user_id for _ in range(num_movies)])
    X_test = {'users': user_list, 'movies':movies_list}
    print('='*100)

    model = Recommender(users=num_users, 
                        movies=num_movies,
                        latent_dim=latent_dim,
                        use_bias=use_bias,
                        global_bias=use_global_bias)
    model.load_weights(filepath=model_path).expect_partial()
    logging.info('successfully loaded trained model weights')

    logging.info('recommending movies for user id %d ...', user_id)
    y_pred = model.predict(x=X_test, batch_size=batch_size)
    result = pd.DataFrame({'ratings': y_pred[:,0]})
    result = movie_names.join(result, on='movie_id', how='inner')

    assert len(result) == num_movies, 'result must be same length as num_movies'

    result.sort_values(by='ratings', ascending=False, inplace=True)
    result.to_csv(args['OUTPUT_FILE'], index=False)
    logging.info('saved recommendations to output file')
    print('='*100)

    top_n_results = result.head(10).values
    for _, year, name, rating in top_n_results:
        print('{:80}({})   {}'.format(name, int(year), rating))

def main():

    args = docopt(__doc__)
    
    if args['train']:
        loggerFile = 'logs/train-{}.log'.format(datetime.now().strftime('%Y%m%d-%H%M%S'))
    elif args['test']:
        loggerFile = 'logs/test-{}.log'.format(datetime.now().strftime('%Y%m%d-%H%M%S'))
    elif args['recommend']:
        loggerFile = 'logs/recommend-{}.log'.format(datetime.now().strftime('%Y%m%d-%H%M%S'))
    else:
        loggerFile = 'logs.logs.log'

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s:%(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(loggerFile,mode="w"),
                            logging.StreamHandler(sys.stdout)
                        ]
                       )

    seed = int(args['--seed'])
    tf.random.set_seed(seed)

    if args['train']:
        logging.info('Train Mode')
        if args['baseline']:
            trainBaseline(args)
        else:
            trainSVD(args)
    elif args['recommend']:
        logging.info('Recommend Mode')
        recommend(args)
    elif args['test']:
        logging.info('Test Mode')
        if args['baseline']:
            testBaseline(args)
        else:
            test(args)
    else:
        logging.error('invalid run mode. expected [train/test/recommend]')
        raise RuntimeError('invalid run mode')
    
if __name__ == '__main__':
    main()