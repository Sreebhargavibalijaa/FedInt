class Config:
    def __init__(self, dropout=0.5, learning_rate=0.001, num_epochs=50, batch_size=32):
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
