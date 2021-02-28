from part1_setup_data_normalization import load_data
from part1_cnn import cnn
from part1_train import train_cnn
from part1_statistics import plot_trainval_loss
from part2_load_model import loadmodel
"""
load dataset
"""
import time
data = load_data()
network = loadmodel(15)

# train = train_cnn(network, data)

#
# print(train.train_loss_history)
# print(train.val_loss_history)
# plot_trainval_loss(train.train_loss_history, train.val_loss_history)

# print("--- %s seconds ---" % (time.time() - start_time))
