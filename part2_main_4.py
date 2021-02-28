
from part1_setup_data_normalization import load_data
from part2_load_model import loadmodel
from part1_train import train_cnn
from part1_statistics import plot_trainval_loss
data = load_data()
network = loadmodel(15)

for param in network.model.parameters():
    param.requires_grad =False

total_trainable_params = sum(
            p.numel() for p in network.model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

for i in range (30):
  for param in network.model.features[i].parameters():
      param.requires_grad = True

train = train_cnn(network.model, data)


print(train.train_loss_history)
print(train.val_loss_history)
plot_trainval_loss(train.train_loss_history, train.val_loss_history)