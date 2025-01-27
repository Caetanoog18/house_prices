import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pre_processing import PreProcessingTest
from tensorflow.keras.models import load_model
from train import Train_NeuralNetwork, Train_RandomForestRegressor, Train_GradientBoostingRegressor


class Test(ABC):
    def __init__(self, pre_processing: PreProcessingTest):
        self.test_data = None
        self.pre_processing = pre_processing

    @abstractmethod
    def apply_model(self):
        pass


class TestNeuralNetwork(Test):
    def __init__(self, pre_processing: PreProcessingTest):
        super().__init__(pre_processing)
        self.train_neural_network = Train_NeuralNetwork().train()

    def apply_model(self):
        with open('database/test_database.pkl', 'rb') as file:
            self.test_data = pickle.load(file)

        neural_network_model = load_model('database/neural_network_model.keras')

        test_predictions = neural_network_model.predict(self.test_data)

        # Removing the warning - RuntimeWarning: overflow encountered in expm1
        test_predictions =  test_predictions.astype(np.float128)

        test_predictions_df = np.expm1(test_predictions).flatten()

        test_ids = pd.read_csv('database/test.csv')['Id']

        submission_file = pd.DataFrame({'Id': test_ids, 'SalePrice': test_predictions_df})
        submission_file.to_csv('database/submission.csv', index=False)

        print("The submission file has been saved")

class TestRandomForestRegressor(Test):
    def __init__(self, pre_processing: PreProcessingTest):
        super().__init__(pre_processing)
        self.train_random_forest_regressor = Train_RandomForestRegressor().train()

    def apply_model(self):
        with open('database/test_database.pkl', 'rb') as file:
            self.test_data = pickle.load(file)

        with open('database/random_forest_model.pkl', 'rb') as file:
            random_forest_model = pickle.load(file)


        test_predictions = random_forest_model.predict(self.test_data)

        # Removing the warning - RuntimeWarning: overflow encountered in expm1
        test_predictions =  test_predictions.astype(np.float128)

        test_predictions_df = np.expm1(test_predictions).flatten()

        test_ids = pd.read_csv('database/test.csv')['Id']

        submission_file = pd.DataFrame({'Id': test_ids, 'SalePrice': test_predictions_df})
        submission_file.to_csv('database/submission.csv', index=False)

        print("The submission file has been saved")

class TestGradientBoostingRegressor(Test):
    def __init__(self, pre_processing: PreProcessingTest):
        super().__init__(pre_processing)
        self.train_gradient_boosting = Train_GradientBoostingRegressor().train()

    def apply_model(self):
        with open('database/test_database.pkl', 'rb') as file:
            self.test_data = pickle.load(file)

        with open('database/random_forest_model.pkl', 'rb') as file:
            xgb_model = pickle.load(file)

        test_predictions = xgb_model.predict(self.test_data)

        # Removing the warning - RuntimeWarning: overflow encountered in expm1
        test_predictions = test_predictions.astype(np.float128)

        test_predictions_df = np.expm1(test_predictions).flatten()

        test_ids = pd.read_csv('database/test.csv')['Id']

        submission_file = pd.DataFrame({'Id': test_ids, 'SalePrice': test_predictions_df})
        submission_file.to_csv('database/submission.csv', index=False)

        print("The submission file has been saved")

