import os

from numpy.core.defchararray import join

_params = {}
_params['DATA_DIRNAME'] = 'data'
_params['OUT_DIRNAME'] = 'output'
_params['USER_MAP'] = 'user.json'
_params['MOVIE_MAP'] = 'movie.json'
_params['BEST_MODEL'] = 'model/best_model'
_params['X_TEST_FILE'] = 'x_test.csv'
_params['Y_TEST_FILE'] = 'y_test.csv'

_params['CURDIR'] = os.getcwd()
_params['DATADIR'] = os.path.join(_params['CURDIR'], _params['DATA_DIRNAME'])
_params['OUTDIR'] = os.path.join(_params['CURDIR'], _params['OUT_DIRNAME'])
_params['CHECKPOINT'] = os.path.join(_params['CURDIR'], _params['OUT_DIRNAME'], _params['BEST_MODEL'])
_params['X_TEST_FILEPATH'] = os.path.join(_params['OUTDIR'], _params['X_TEST_FILE'])
_params['Y_TEST_FILEPATH'] = os.path.join(_params['OUTDIR'], _params['Y_TEST_FILE'])

def set(param_name, param_value):
    _params[param_name] = param_value
    if param_name == 'CURDIR':
        _params['DATADIR'] = os.path.join(_params['CURDIR'], _params['DATA_DIRNAME'])
        _params['OUTDIR'] = os.path.join(_params['CURDIR'], _params['OUT_DIRNAME'])
        _params['CHECKPOINT'] = os.path.join(_params['CURDIR'], _params['OUT_DIRNAME'], _params['BEST_MODEL'])
        _params['X_TEST_FILEPATH'] = os.path.join(_params['OUTDIR'], _params['X_TEST_FILE'])
        _params['Y_TEST_FILEPATH'] = os.path.join(_params['OUTDIR'], _params['Y_TEST_FILE'])

def get(param_name):
    return _params[param_name]