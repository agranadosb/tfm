import os.path
import random
from abc import ABC, abstractmethod
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from tfm.constants import LIGHTS_DATASET
from tfm.constants.types import State, Sample
from tfm.data.lights import LightsOutDataset
from tfm.data.puzzle import Puzzle8MnistDataset


class BaseGenerator(ABC):
    """This class is in charge to provide an interface to generate datasets.
    The basis of this model is on the concepts of "State", "Sample" and
    "Sequence".

    A "State" is the representation of a state of a problem. It can adopt any
    form like a matrix, a vector, a string, etc.

    A "Sample" is a tuple of a "State" and a sequence of "States". The first
    element of the tuple is the "State" and the second element is a sequence
    of "States". This sequence of "States" are the result of applying actions
    to the first "State".

    For example, suppose that we are working with the
    rock-paper-scissors game. A "State" could be a string with the value
    "rock". A "Sample" could be ("rock", ("paper", "scissors", "rock")). As we
    can see, the sequence of states is the result of applying all the possible
    actions of the problem to the first state. Each action is represented by
    the index of the sequence of states. For example, here the index 0 is
    "paper", the index 1 is "scissors" and the index 2 is "rock".

    In some case an action could not be applied to a state. For example, in the
    tick-tack-toe game, if a player has already put a piece in a position, it
    can't put another piece in the same position. In this case, the action on
    index "n" of the sequence of states will be "None". On the tick-tack-toe
    example, if the first state is "X" and the action "put a piece in the
    position 0" is applied, the result will be "None" because the position 0
    is already occupied. The sequence of states will be:

    (None, ...).

    A "Sequence" is a sequence of "Samples". When a "Sequence" is generated,
    it always starts with the initial state of the problem. Then, it applies
    actions to the initial state and generates a "Sample". This "Sample" is
    added to the "Sequence". Then, the "Sample" is used to select a new state
    and the process is repeated until the "Sequence" is completed. If it's
    required, the "Sequence" can be shuffled.

    All the "Sequences" are generated in the same way. All the "Sequences" have
    the same length and the same number of "Samples".

    To do so it's necessary to implement the following methods:

    - init_state: Returns the initial state used to generate a "Sequence".
    - select: Given a "Sample", it selects a "State" from the sequence of
    "States" of the "Sample".
    - move: Given a "State", it returns a "Sample" of the given "State".
    - save_sample: Given a path and a "Sample", it saves the "Sample" in the path.

    Parameters
    ----------
    sequences : int
        Number of sequences to generate.
    sequence_length : int
        Length of the sequences to generate.
    shuffle : bool, optional
        If True, the sequences are shuffled, by default True.
    """

    def __init__(self, sequences: int, sequence_length: int, shuffle: bool = True):
        self.indices = None
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.shuffle = shuffle

    @abstractmethod
    def init_state(self) -> State:
        """
        Returns the initial state used to generate a "Sequence". This method
        is called when the method "generate" is called. Each "Sequence" starts
        with the initial state.

        Examples
        --------
        >>> generator = ClassInheritingBaseGenerator(sequences=1, sequence_length=2)
        >>> generator.init_state()
        'state'

        Returns
        -------
        State
            The initial state.
        """
        ...

    @abstractmethod
    def select(self, sample: Sample) -> State:
        """
        Given a "Sample", it selects a "State" from the sequence of "States" of
        the "Sample".

        Parameters
        ----------
        sample : Sample

        Examples
        --------
        >>> generator = ClassInheritingBaseGenerator(sequences=1, sequence_length=2)
        >>> generator.select(('state', ('new-state-1', 'new-state-2', 'new-state-3')))
        'new-state-1'

        >>> generator.select(('state', ('new-state-1', None, None)))
        'new-state-1'

        Returns
        -------
        State
            The selected "State".
        """
        ...

    @abstractmethod
    def move(self, state: State) -> Sample:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Given a "State", it returns a "Sample" of the given "State". The
        "Sample" is a tuple of a "State" and a sequence of "States". Each
        element of the sequence of "States" is the result of applying an action
        to the initial "State". All the actions must be predefined. If an
        action can't be applied to the initial "State", the action will be
        "None".
        
        Parameters
        ----------
        state : State
            The initial "State".
        
        Examples
        --------
        >>> generator = ClassInheritingBaseGenerator(sequences=1, sequence_length=2)
        >>> generator.move('state')
        ('state', ('new-state-1', 'new-state-1', 'new-state-3'))
        
        >>> generator.move('state2')
        ('state2', ('new-state-1', None, 'new-state-3'))

        Returns
        -------
        Sample
            The "Sample" of the given "State".
        """
        ...

    @abstractmethod
    def save_sample(self, sample: Sample, path: str, index: int) -> "BaseGenerator":
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Save a "Sample" in the given path. The "Sample" will be saved in a
        folder whose name is the given index. This folder have to contain a
        file named "State" with the state of the sample and a folder named
        actions, with a file per each possible action applied to the state.
        Each file on the folder with actions will be named with a number from
        0 to the number of possible actions minus one. If one action is not
        possible, the file with the number representing the action will not be
        created.

        Parameters
        ----------
        sample : Sample
            The "Sample" to save.
        path : str
            The path where the "Sample" will be saved.
        index : int
            The index of the "Sample".

        Examples
        --------
        >>> generator = ClassInheritingBaseGenerator(sequences=1, sequence_length=2)
        >>> generator.save_sample(
        ...     ('state', ('new-state-1', 'new-state-2', 'new-state-3')), 'dataset', 0
        ... )
        >>> os.listdir('dataset')
        ['0']

        >>> os.listdir('dataset/0')
        ['state', 'actions']

        >>> os.listdir('dataset/0/actions')
        ['0', '1', '2']

        >>> generator.save_sample(
        ...     ('state', ('new-state-1', None, 'new-state-3')), 'dataset', 1
        ... )
        >>> os.listdir('dataset')
        ['0', '1']

        >>> os.listdir('dataset/1')
        ['state', 'actions']

        >>> os.listdir('dataset/1/actions')
        # The "new-state-2" is not possible so the file is not created
        ['0', '2']
        """
        ...

    def save(self, path: str, sample: list[Sample]) -> "BaseGenerator":
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Save a list of "Samples" in the given path. Each "Sample" will be
        saved in a folder whose name is an index. This index is obtained from
        the "indices" attribute. This attribute is a list of indices. This
        list is initialized when the method "generate" is called. Each time a
        "Sample" is saved, the index is removed from the list.

        Parameters
        ----------
        path : str
            The path where the "Samples" will be saved.
        sample : list[Sample]
            The list of "Samples" to save.

        Examples
        --------
        >>> generator = ClassInheritingBaseGenerator(sequences=1, sequence_length=2)
        >>> generator.indices = [3, 2, 1, 0]
        >>> generator.save(
        ...     'dataset', [('state', ('new-state-1', 'new-state-2', 'new-state-3'))]
        ... )
        >>> os.listdir('dataset')
        ['0']

        >>> os.listdir('dataset/0')
        ['state', 'actions']

        >>> os.listdir('dataset/0/actions')
        ['0', '1', '2']

        >>> generator.indices
        [3, 2, 1]

        >>> generator.save('dataset', [('state', ('new-state-1', None, 'new-state-3'))])
        >>> os.listdir('dataset')
        ['0', '1']

        >>> os.listdir('dataset/1')
        ['state', 'actions']

        >>> os.listdir('dataset/1/actions')
        # The "new-state-2" is not possible so the file is not created
        ['0', '2']

        >>> generator.indices
        [3, 2]
        """
        for state, states in sample:
            index = self.indices.pop()
            self.save_sample(state, path, self.indices.pop())

            state_file = Path(path) / str(index) / "state"
            actions_folder = Path(path) / str(index) / "actions"
            if not (state_file.exists() and actions_folder.exists()):
                raise ValueError(
                    f"The sample {index} is not saved correctly. "
                    f"The state file or the actions folder doesn't exist."
                )

        return self

    def sequence(self, shuffle: bool = True) -> list[Sample]:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Generate a sequence of "Samples". The sequence always starts with the
        initial state of the problem. Then, it applies actions to the initial
        state and generates a "Sample". This "Sample" is added to the
        "Sequence". Then, the "Sample" is used to select a new state and the
        process is repeated until the "Sequence" is completed. If it's
        required, the "Sequence" can be shuffled.

        Parameters
        ----------
        shuffle : bool, optional
            If True, the sequences are shuffled, by default True.

        Examples
        --------
        >>> generator = ClassInheritingBaseGenerator(sequences=1, sequence_length=2)
        >>> generator.sequence()
        # An action can be None if it's not possible to apply it to the state,
        # this is the case of the action 1 on the second sample
        [
            ('state', ('new-state-1', 'new-state-2', 'new-state-3')),
            ('state', ('new-state-1', None, 'new-state-3'))
        ]

        Returns
        -------
        list[Sample]
            The sequence of "Samples".
        """
        current_state = self.init_state()
        sequence = []

        for _ in range(self.sequence_length * 4):
            sample = self.move(current_state)
            current_state = self.select(sample)
            sequence.append(sample)

        if shuffle:
            random.shuffle(sequence)

        return sequence[:self.sequence_length]

    def generate(self, path: str, shuffle: bool = True) -> "BaseGenerator":
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Generate a dataset of "Samples" and save it in the given path. Each
        "Sample" will be saved in a different folder. Each folder will be named
        with a number from 0 to the number of "Samples" minus one. Each folder
        will contain a file named "State" with the state of the sample and a
        folder named actions, with a file per each possible action applied to
        the state. Each file on the folder with actions will be named with a
        number from 0 to the number of possible actions minus one. If one
        action is not possible, the file with the number representing the
        action will not be created.

        Parameters
        ----------
        path : str
            The path where the dataset will be saved.
        shuffle : bool, optional
            If True, the sequences are shuffled, by default True.

        Examples
        --------
        >>> generator = ClassInheritingBaseGenerator(sequences=1, sequence_length=2)
        >>> generator.generate('dataset')
        >>> os.listdir('dataset')
        ['0', '1']

        >>> os.listdir('dataset/0')
        ['state', 'actions']

        >>> os.listdir('dataset/0/actions')
        ['0', '1', '3'] # The action 2 is not possible
        """
        self.indices = [i for i in range(self.sequences * self.sequence_length)]
        self.indices.reverse()

        if shuffle:
            random.shuffle(self.indices)

        for i in range(self.sequences):
            self.save(path, self.sequence(shuffle=shuffle))

        return self


class DataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size: int, input_size: int, num_workers: int, dataset: str
    ):
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.dataset_size = self.input_size
        self.num_workers = num_workers

        self.dataset_class = Puzzle8MnistDataset
        if dataset == LIGHTS_DATASET:
            self.dataset_size = 5
            self.dataset_class = LightsOutDataset

        self.training = None
        self.evaluation = None

    def setup(self, stage: str):
        transformations = torch.nn.Sequential(
            transforms.Resize((self.input_size, self.input_size)),
            transforms.RandomResizedCrop(
                (self.input_size, self.input_size), scale=(0.95, 1.0)
            ),
            transforms.RandomRotation(15),
            transforms.GaussianBlur(3),
        )

        self.training = self.dataset_class(
            self.dataset_size, 100, self.batch_size, transformations=transformations
        )
        self.evaluation = self.dataset_class(self.dataset_size, 16, self.batch_size)

    def train_dataloader(self):
        return DataLoader(
            self.training, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.evaluation, batch_size=self.batch_size, num_workers=self.num_workers
        )
