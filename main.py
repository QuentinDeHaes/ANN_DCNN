from loader import load_data
from model import cnn
from train import train_cnn
from cnnplot import plot_trainval_loss

data = load_data(None)
network = cnn()
train = train_cnn(network, data)

print(train.train_loss_history)
print(train.val_loss_history)
plot_trainval_loss(train.train_loss_history, train.val_loss_history)