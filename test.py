import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


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

        # Removing the warning - RuntimeWarning: overflow encountered in expm1
        test_predictions =  test_predictions.astype(np.float128)

        test_predictions_df = np.expm1(test_predictions).flatten()

        test_ids = pd.read_csv('database/test.csv')['Id']

        submission_file = pd.DataFrame({'Id': test_ids, 'SalePrice': test_predictions_df})
        submission_file.to_csv('database/submission.csv', index=False)

        print("The submission file has been saved")

