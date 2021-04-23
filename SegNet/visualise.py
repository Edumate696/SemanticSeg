import random

from matplotlib import pyplot as plt

plt.style.use("ggplot")


def plot_random_sample(X_train, y_train):
    # Visualize any random image along with the mask
    ix = random.randint(0, len(X_train))
    has_mask = y_train[ix].max() > 0  # salt indicator

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))

    ax1.imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
    if has_mask:  # if salt
        # draw a boundary(contour) in the original image separating salt and non-salt areas
        ax1.contour(y_train[ix].squeeze(), colors='k', linewidths=5, levels=[0.5])
    ax1.set_title('Seismic')

    ax2.imshow(y_train[ix].squeeze(), cmap='gray', interpolation='bilinear')
    ax2.set_title('Salt')
