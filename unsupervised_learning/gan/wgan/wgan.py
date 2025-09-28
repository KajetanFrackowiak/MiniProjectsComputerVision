from tensorflow import nn
from keras import layers, Model

class Generator(Model):
    def __init__(self, img_dim=32*32*3):
        super(Generator, self).__init__()
        self.fc1 = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(256, activation="relu")
        self.fc3 = layers.Dense(img_dim, activation="tanh")
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class Critic(Model):
    def __init__(self, img_dim=32*32*3):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(256)
        self.fc2 = layers.Dense(128)
        self.fc3 = layers.Dense(1)

    def call(self, x):
        x = self.fc1(x)
        x = nn.leaky_relu(x, alpha=0.2)
        
        x = self.fc2(x)
        x = nn.leaky_relu(x, alpha=0.2)

        x = self.fc3(x)

        return x