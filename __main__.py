from torchvision import transforms

from tfm.data.puzzle import Puzzle8MnistGenerator
from matplotlib import pyplot as plt


if __name__ == "__main__":
    p = Puzzle8MnistGenerator()
    for i in range(10):
        plt.imshow(transforms.ToPILImage()(p.get(ordered=True)[0]))
        plt.show()
