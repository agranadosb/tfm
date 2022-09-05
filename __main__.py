import time

from torchvision import transforms

from tfm.data.puzzle import Puzzle8MnistGenerator
from matplotlib import pyplot as plt

from tfm.plots.images import plot_images

if __name__ == "__main__":
    init = time.time()
    p = Puzzle8MnistGenerator()
    print(f"Create object: {time.time() - init}")
    init = time.time()
    images = [transforms.ToPILImage()(p.get()[0]) for _ in range(10)]
    print(f"Load images: {time.time() - init}")
    plot_images(images)
