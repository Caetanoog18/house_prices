from train import Train
from pre_processing import PreProcessing, Analyze

analyze = Analyze()
analyze.distribution_graph()
analyze.correlation_features()

test = PreProcessing(analyze)
test.pre_processing_train()
test.pre_processing_test()

# train = Train()
# train.train_xgb()







