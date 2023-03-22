import time

import torch
import torchvision
from torchvision import transforms

from tfm.data.puzzle import Puzzle8MnistGenerator
from tfm.model.trainer import UnetMultiModalTrainer, Configuration

from tfm.plots.images import plot_images
from tfm.utils.puzzle import is_solvable

if __name__ == "__main__":
    init = time.time()
    p = Puzzle8MnistGenerator(order=[1, 2, 3, 8, 0, 4, 7, 6, 5])
    """print(f"Create object: {time.time() - init}")
    init = time.time()
    images = [transforms.ToPILImage()(p.get()[0]) for _ in range(10)]
    print(f"Load images: {time.time() - init}")
    init = time.time()
    plot_images(images)
    print(f"Plot images: {time.time() - init}")
    init = time.time()
    order = [1, 2, 3, 8, 0, 4, 7, 6, 5]
    sequence_result, movements_result = p.random_movements()
    print(f"Random movements: {time.time() - init}")
    init = time.time()
    result = is_solvable(sequence_result, order)
    print(f"Check solvable: {time.time() - init}")

    image, order = p.get(movements=2)
    print(image.size())
    print(order)"""

    # images, transitions, movements = p.get_batch(9)

    # print(movements)
    # plot_images([transforms.ToPILImage()(image) for image in images])
    # plot_images([transforms.ToPILImage()(a) for a in transitions])

    UnetMultiModalTrainer(Configuration(64, 100)).train(100)

    """images, transitions, movements = p.get_batch(32)

    plot_images([transforms.ToPILImage()(image) for image in images])
    plot_images([transforms.ToPILImage()(a) for a in transitions])
    model = UNet(1)

    images = torchvision.transforms.Resize((64, 64))(images)
    prediction = model(images)
    prediction = torchvision.transforms.Resize((28 * 3, 28 * 3))(prediction)
    images = [transforms.ToPILImage()(image) for image in prediction[:9]]
    plot_images(images)"""
