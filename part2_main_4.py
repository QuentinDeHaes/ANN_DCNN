from part1_setup_data_normalization import load_data
from part2_load_model import loadmodel
from part1_train import train_cnn
from part1_statistics import plot_trainval_loss

data = load_data()
network = loadmodel(15)

for i in range(len(network.model.classifier)):
    for param in network.model.classifier[i].parameters():
        param.requires_grad = False

train = train_cnn(network.model, data)

print(train.train_loss_history)
print(train.val_loss_history)
plot_trainval_loss(train.train_loss_history, train.val_loss_history)
