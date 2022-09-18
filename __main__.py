import time

from torchvision import transforms

from tfm.data.puzzle import Puzzle8MnistGenerator

from tfm.plots.images import plot_images

if __name__ == "__main__":
    init = time.time()
    p = Puzzle8MnistGenerator(order=[1, 2, 3, 8, 0, 4, 7, 6, 5])
    print(f"Create object: {time.time() - init}")
    init = time.time()
    images = [transforms.ToPILImage()(p.get()[0]) for _ in range(10)]
    print(f"Load images: {time.time() - init}")
    init = time.time()
    plot_images(images)
    print(f"Plot images: {time.time() - init}")
    init = time.time()
    order = [1, 2, 3, 8, 0, 4, 7, 6, 5]
    sequence_result, movements_result = p._random_movements()
    print(f"Random movements: {time.time() - init}")
