from train import Train_NeuralNetwork
from pre_processing import PreProcessing, PreProcessingTest, PreProcessingTrain, Analyze
from test import TestNeuralNetwork, TestRandomForestRegressor, TestGradientBoostingRegressor

train = PreProcessingTrain().pre_processing()
test_pre_processing = PreProcessingTest().pre_processing()
model = TestNeuralNetwork(test_pre_processing).apply_model()








