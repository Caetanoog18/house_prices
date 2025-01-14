from train import Train
from pre_processing import PreProcessing

test = PreProcessing()
test.pre_processing_train()
test.pre_processing_test()

train = Train()
train.train()



