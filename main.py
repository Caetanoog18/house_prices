from train import Train
from pre_processing import PreProcessing, Graph

test = PreProcessing()
test.pre_processing_train()
test.pre_processing_test()

# train = Train()
# train.train_xgb()

graph = Graph(test)
graph.distribution_graph()
graph.correlation_features()





