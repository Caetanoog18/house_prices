import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score


class Test:
    def apply_model(self):
        with open('database/test_database.pkl', 'rb') as file:
            test_data = pickle.load(file)

        # with open('database/random_forest_model.pkl', 'rb') as file:
        #     neural_network_model = pickle.load(file)
        #
        # with open('database/random_forest_model.pkl', 'rb') as file:
        #     neural_network_model = pickle.load(file)

        neural_network_model = load_model('database/neural_network_model.keras')

        test_predictions = neural_network_model.predict(test_data)

        test_predictions_df = np.expm1(test_predictions).flatten()

        test_ids = pd.read_csv('database/test.csv')['Id']

        submission_file = pd.DataFrame({'Id': test_ids, 'SalePrice': test_predictions_df})
        submission_file.to_csv('database/submission.csv', index=False)

        print("The submission file has been saved")

