import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


class Analyze:
    def __init__(self):
        self.train_database = pd.read_csv('database/train.csv')
        self.test_database = pd.read_csv('database/test.csv')
        self.important_features = {}

    def distribution_graph(self):
        plt.figure(figsize=(10, 10))
        plt.hist(self.train_database['SalePrice'], bins=30, color='red', edgecolor='black')
        plt.xlabel('Sale Price')
        plt.ylabel('Count')
        plt.title('Distribution of house price')
        plt.show()

    def correlation_features(self):
        numeric_data = self.train_database.select_dtypes(include=['number'])
        correlation_matrix = numeric_data.corr()
        sale_price_correlation = correlation_matrix['SalePrice'].sort_values(ascending=False)
        print(sale_price_correlation)

        sale_price_correlation = sale_price_correlation.drop(columns='SalePrice')

        plt.figure(figsize=(10, 8))
        sale_price_correlation.plot(kind='bar', color='red')
        plt.xlabel('Features')
        plt.ylabel('Correlation')
        plt.show()

        for feature, value in sale_price_correlation.items():
            if value > 0.50:
                self.important_features.update({feature: value})

        print(self.important_features.keys())

    def boxplot_graph(self):
        self.correlation_features()
        for feature in self.important_features.keys():
            graph = px.box(self.train_database, y=feature)
            # graph.show()


class PreProcessing:
    def __init__(self, analyze: Analyze):
        self.analyze = analyze
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.scaler = None

    def pre_processing_train(self):
        # (1460, 81)
        # print(self.analyze.train_database.shape)
        # (1459, 80)
        # print(self.analyze.test_database.shape)

        train_data = self.analyze.train_database.loc[:, ['LotShape','SalePrice', 'OverallQual', 'GrLivArea',
                                                         'GarageCars', 'GarageArea', 'TotalBsmtSF']]

        # print(train_data.head())
        # print(self.X_train.shape)
        # print(self.y_train.shape)


        label_encoder_lot_shape = LabelEncoder()
        train_data['LotShape'] = label_encoder_lot_shape.fit_transform(train_data['LotShape'])


        columns = ['LotShape','SalePrice', 'OverallQual', 'GrLivArea',
                                                         'GarageCars', 'GarageArea', 'TotalBsmtSF',]

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
            else:
                print(f'No missing values in column {column}')

        #train_data.fillna(train_data['LotFrontage'].mean(), inplace=True)

        # Test inconsistent values
        #print(train_data.loc[pd.isnull(train_data['LotFrontage'])])

        # Removing outliers from target variable SalePrice
        train_data['SalePrice'] = np.log1p(train_data['SalePrice'])

        # Dropping the SalesPrice Column
        x = train_data.drop(columns=['SalePrice'])
        y = train_data['SalePrice']

        self.scaler = MinMaxScaler()
        #scaler = StandardScaler()
        x = self.scaler.fit_transform(x)

        self.X_train, self.y_train, self.X_val, self.y_val = train_test_split(x, y, test_size=0.25, random_state=42)

        with open('database/train_database.pkl', 'wb') as file:
            pickle.dump([self.X_train, self.y_train, self.X_val, self.y_val], file)


    def pre_processing_test(self):
        test_data = self.analyze.test_database.loc[:, ['LotShape', 'OverallQual', 'GrLivArea',
                                                         'GarageCars', 'GarageArea', 'TotalBsmtSF',]]

        # print(test_data.shape) # (1459, 8)

        label_encoder_lot_shape = LabelEncoder()
        test_data['LotShape'] = label_encoder_lot_shape.fit_transform(test_data['LotShape'])


        columns = ['LotShape', 'OverallQual', 'GrLivArea',
                                                         'GarageCars', 'GarageArea', 'TotalBsmtSF',]

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

        test_data.fillna(test_data['TotalBsmtSF'].mean(), inplace=True)
        test_data.fillna(test_data['GarageCars'].mean(), inplace=True)
        test_data.fillna(test_data['GarageArea'].mean(), inplace=True)


        # Test inconsistent values
        print(test_data.loc[pd.isnull(test_data['TotalBsmtSF'])])
        print(test_data.loc[pd.isnull(test_data['GarageCars'])])
        print(test_data.loc[pd.isnull(test_data['GarageArea'])])
        print(test_data.loc[pd.isnull(test_data['TotalBsmtSF'])])

        if self.scaler is not None:
            test_data_scaled = self.scaler.transform(test_data)
        else:
            raise ValueError("Scaler not initialized. Ensure you call pre_processing_train() first.")

        with open('database/test_database.pkl', 'wb') as file:
            pickle.dump(test_data_scaled, file)







