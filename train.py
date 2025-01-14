import pickle
from pre_processing import PreProcessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


class Train:
    def __init__(self):
        self.X_training = None
        self.y_training = None
        self.X_val = None
        self.y_val = None

    def train(self):
        with open('database/train_database.pkl', 'rb') as file:
            self.X_training, self.X_val, self.y_training, self.y_val = pickle.load(file)

        print('X_training shape', self.X_training.shape)
        print('y_training shape', self.y_training.shape)
        print('X_val shape', self.X_val.shape)
        print('y_val shape', self.y_val.shape, '\n')

        xgb_model = GradientBoostingRegressor()
        xgb_model.fit(self.X_training, self.y_training)

        xgb_prediction = xgb_model.predict(self.X_val)

        r2_score_value = r2_score(self.y_val, xgb_prediction)
        mse_value = mean_squared_error(self.y_val, xgb_prediction)


        residuals = self.y_val - xgb_prediction
        print(residuals.describe())


        print('r2_score =', r2_score_value)
        print('mse_value =', mse_value)







