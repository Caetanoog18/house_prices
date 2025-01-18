from test import Test
from train import Train
from pre_processing import PreProcessing, Analyze

analyze = Analyze()
analyze.distribution_graph()
analyze.boxplot_graph()

test = PreProcessing(analyze)
test.pre_processing_train()
test.pre_processing_test()

train = Train()
# train.train_xgb()
# train.train_random_forest()
train.neural_network()


test = Test()
test.apply_model()








