

# Superclass for models
class FNCModel(object):

    def __init__(self, name, args):
        self.args = args
        self.name = name

    # Train model
    def train(self, data, train_args):
        pass

    # Use model to predict
    def predict(self, data, predict_args):
        pass

    # Save model to disk
    def save(self, path):
        pass
