from part1_setup_data_normalization import load_data
from part2_load_model import loadmodel
from part1_train import train_cnn
from part1_statistics import plot_trainval_loss

data = load_data()
network = loadmodel(15)
ct = 0

# source : https://stackoverflow.com/questions/62523912/how-to-freeze-selected-layers-of-a-model-in-pytorch
for i in range (15,30):
  for param in network.model.features[i].parameters():
      param.requires_grad = False

train = train_cnn(network.model, data)


print(train.train_loss_history)
print(train.val_loss_history)
plot_trainval_loss(train.train_loss_history, train.val_loss_history)