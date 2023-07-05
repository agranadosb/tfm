from tfm.model.base import ConvBlock, ResidualBlock

ORDERED_ORDER = [1, 2, 3, 8, 0, 4, 7, 6, 5]

LEFT_RIGHT_MOVEMENTS = [1, -1]
TOP_BOT_MOVEMENTS = [3, -3]
MOVEMENTS = [*LEFT_RIGHT_MOVEMENTS, *TOP_BOT_MOVEMENTS]

LABEL_TO_STRING = {
    "puzzle8": {
        0: "Right",
        1: "Left",
        2: "Bottom",
        3: "Top",
    },
    "lights-out": {i: f"Light {i}" for i in range(25)},
}
MOVEMENT_TO_LABEL = {
    "puzzle8": {1: 0, -1: 1, 3: 2, -3: 3},
    "lights-out": {i: i for i in range(25)},
}
LABEL_TO_MOVEMENT = {
    "puzzle8": {value: key for key, value in MOVEMENT_TO_LABEL["puzzle8"].items()},
    "lights-out": {value: key for key, value in MOVEMENT_TO_LABEL["lights-out"].items()},
}

BLOCKS = {"conv": ConvBlock, "residual": ResidualBlock}
