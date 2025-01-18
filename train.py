import os
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

class Train:
    def __init__(self):
        self.X_training = None
        self.y_training = None
        self.X_val = None
        self.y_val = None

    def train_xgb(self):
        if os.path.exists('database/train_database.pkl'):
            with open('database/train_database.pkl', 'rb') as file:
                self.X_training, self.X_val, self.y_training, self.y_val = pickle.load(file)
        else:
            raise FileNotFoundError('Train database was not found.', '\n')

        print('X_training shape', self.X_training.shape)
        print('y_training shape', self.y_training.shape)
        print('X_val shape', self.X_val.shape)
        print('y_val shape', self.y_val.shape, '\n')

        xgb_model = GradientBoostingRegressor()
        xgb_model.fit(self.X_training, self.y_training)

        with open('database/xgb_model.pkl', 'wb') as file:
            pickle.dump(xgb_model, file)

        xgb_prediction = xgb_model.predict(self.X_val)

        r2_score_value = r2_score(self.y_val, xgb_prediction)
        mse_value = mean_squared_error(self.y_val, xgb_prediction)


        residuals = self.y_val - xgb_prediction
        print(residuals.describe())


        print('r2_score =', r2_score_value)
        print('mse_value =', mse_value, '\n')

    def train_random_forest(self):
        if os.path.isfile('database/train_database.pkl'):
            with open('database/train_database.pkl', 'rb') as file:
                self.X_training, self.X_val, self.y_training, self.y_val = pickle.load(file)
        else:
            raise FileNotFoundError('Train database was not found.', '\n')

        print('X_training shape', self.X_training.shape)
        print('y_training shape', self.y_training.shape)
        print('X_val shape', self.X_val.shape)
        print('y_val shape', self.y_val.shape, '\n')

        random_forest = RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=42)
        random_forest.fit(self.X_training, self.y_training)

        with open('database/random_forest_model.pkl', 'wb') as file:
            pickle.dump(random_forest, file)

        predictions = random_forest.predict(self.X_val)
        r2_score_value = r2_score(self.y_val, predictions)
        mse_value = mean_squared_error(self.y_val, predictions)

        print('r2_score =', r2_score_value)
        print('mse_value =', mse_value, '\n')

    def neural_network(self):
        if os.path.exists('database/train_database.pkl'):
            with open('database/train_database.pkl', 'rb') as file:
                self.X_training, self.X_val, self.y_training, self.y_val = pickle.load(file)
        else:
            raise FileNotFoundError('Train database was not found.', '\n')

        print('X_training shape', self.X_training.shape)
        print('y_training shape', self.y_training.shape)
        print('X_val shape', self.X_val.shape)
        print('y_val shape', self.y_val.shape, '\n')

        model = Sequential([
            Dense(128, kernel_initializer='normal',activation='relu', input_shape=(self.X_training.shape[1], )),
            Dense(64, kernel_initializer='normal',activation='relu'),
            Dense(1),
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])


        history = model.fit(self.X_training, self.y_training, validation_data=(self.X_val, self.y_val), epochs=50,
                            batch_size=16, verbose=1)

        model.save('database/neural_network_model.keras')

        predictions = model.predict(self.X_val)
        r2_score_value = r2_score(self.y_val, predictions)
        mse_value = mean_squared_error(self.y_val, predictions)

        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        print('r2_score =', r2_score_value)
        print('mse_value =', mse_value, '\n')












