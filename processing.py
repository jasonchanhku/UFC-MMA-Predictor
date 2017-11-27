# This script does the data mining and machine learning

import pandas as pd
from sklearn.neural_network import MLPClassifier
import pickle


# Function to mine the data


def get_data():
    b = pd.read_csv(
        'https://raw.githubusercontent.com/jasonchanhku/UFC-MMA-Predictor/master/Datasets/Cleansed_Data.csv')

    return b


def training():
    fights_db = get_data()

    best_cols = ['SLPM_delta', 'SAPM_delta', 'STRD_delta', 'TD_delta', 'Odds_delta']

    all_X = fights_db[best_cols]
    all_y = fights_db['Label']

    mlp = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
                        beta_2=0.999, early_stopping=False, epsilon=1e-08,
                        hidden_layer_sizes=(5, 5), learning_rate='constant',
                        learning_rate_init=0.001, max_iter=200, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                        solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
                        warm_start=False)
    mlp.fit(all_X, all_y)

    filename = 'NeuralNet.sav'

    pickle.dump(mlp, open(filename, 'wb'))

    return filename


def get_prediction(df):

    filename = training()

    loaded_model = pickle.load(open(filename, 'rb'))

    result = loaded_model.predict_proba(df)

    return result

best_cols = ['SLPM_delta', 'SAPM_delta', 'STRD_delta', 'TD_delta', 'Odds_delta']

test = get_data()
test2 = test[best_cols].loc[10]
print(get_prediction(test2.reshape(1, -1)))

