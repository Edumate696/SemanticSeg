import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import SegNet
from SegNet.model import Model

if __name__ == '__main__':
    loader: SegNet.DataLoader = SegNet.DataLoader()
    X_train, X_valid, y_train, y_valid = loader.load_dataset()
    # SegNet.plot_random_sample(X_train, y_train)

    model: SegNet.SegNetModel = SegNet.SegNetModel()
    model.load_weights()
    # model.summary()
    # model.fit(X_train, y_train, val_data=(X_valid, y_valid))
    model: Model = model.get_model()

    print(model.evaluate(X_valid, y_valid))
