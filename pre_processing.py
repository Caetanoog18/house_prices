import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class PreProcessing:
    def __init__(self):
        self.train_database = pd.read_csv('database/train.csv')
        self.test_database = pd.read_csv('database/test.csv')
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def pre_processing_train(self):
        # (1460, 81)
        # print(self.train_database.shape)
        # (1459, 80)
        # print(self.test_database.shape)

        train_data = self.train_database.loc[:, ['LotArea', 'LotFrontage', 'LotShape','YearBuilt', 'OverallQual', 'TotalBsmtSF', 'GarageCars', 'SalePrice', 'GrLivArea']]

        # print(train_data.head()) # - 5 rows x 9 Columns

        # print(self.X_train.shape) # - (1460, 8)
        # print(self.y_train.shape) # - (1460,)


        label_encoder_lot_shape = LabelEncoder()
        train_data['LotShape'] = label_encoder_lot_shape.fit_transform(train_data['LotShape'])

        columns = ['LotArea', 'LotFrontage', 'LotShape', 'YearBuilt', 'OverallQual', 'TotalBsmtSF',
                             'GarageCars', 'SalePrice','GrLivArea']

        inconsistencies = []
        # Verifying inconsistent values
        for column in columns:
            negative_values = train_data[train_data[column]<0]
            if not negative_values.empty:
                inconsistencies.append(
                    {'Column': column, 'Number of inconsistencies': len(negative_values), 'Data': negative_values})

            else:
                print(f'No inconsistencies found at column {column}')
        print()
        # Verifying missing values
        missing_values = train_data.isnull().sum()
        print('Missing values by column:')
        for column in columns:
            num_missing = missing_values[column]
            if num_missing>0:
                print(f"Column: {column}, Missing values: {num_missing}")
                print(train_data[train_data[column].isnull()])

        train_data.fillna(train_data['LotFrontage'].mean(), inplace=True)

        # Test inconsistent values
        print(train_data.loc[pd.isnull(train_data['LotFrontage'])])

        # Removing outliers from target variable SalePrice
        train_data['SalePrice'] = np.log1p(train_data['SalePrice'])

        # Dropping the SalesPrice Column
        x = train_data.drop(columns=['SalePrice'])
        y = train_data['SalePrice']

        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        self.X_train, self.y_train, self.X_val, self.y_val = train_test_split(x, y, test_size=0.25, random_state=42)

        with open('database/train_database.pkl', 'wb') as file:
            pickle.dump([self.X_train, self.y_train, self.X_val, self.y_val], file)


    def pre_processing_test(self):
        test_data = self.test_database.loc[:,
                    ['LotArea', 'LotFrontage', 'LotShape', 'YearBuilt', 'OverallQual', 'TotalBsmtSF', 'GarageCars',
                     'GrLivArea']]

        # print(test_data.shape) # (1459, 8)

        label_encoder_lot_shape = LabelEncoder()
        test_data['LotShape'] = label_encoder_lot_shape.fit_transform(test_data['LotShape'])

        columns = ['LotArea', 'LotFrontage', 'LotShape', 'YearBuilt', 'OverallQual', 'TotalBsmtSF',
                             'GarageCars','GrLivArea']

        inconsistencies = []
        # Verifying inconsistent values
        for column in columns:
            negative_values = test_data[test_data[column] < 0]
            if not negative_values.empty:
                inconsistencies.append(
                    {'Column': column, 'Number of inconsistencies': len(negative_values), 'Data': negative_values})

            else:
                print(f'No inconsistencies found at column {column}')
        print()
        # Verifying missing values
        missing_values = test_data.isnull().sum()
        print('Missing values by column:')
        for column in columns:
            num_missing = missing_values[column]
            if num_missing > 0:
                print(f"Column: {column}, Missing values: {num_missing}")
                print(test_data[test_data[column].isnull()])

        test_data.fillna(test_data['LotFrontage'].mean(), inplace=True)
        test_data.fillna(test_data['TotalBsmtSF'].mean(), inplace=True)
        test_data.fillna(test_data['GarageCars'].mean(), inplace=True)

        # Test inconsistent values
        print(test_data.loc[pd.isnull(test_data['LotFrontage'])])
        print(test_data.loc[pd.isnull(test_data['TotalBsmtSF'])])
        print(test_data.loc[pd.isnull(test_data['GarageCars'])])


