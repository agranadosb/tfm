import numpy as np
import torch
from torch import Tensor

from tfm.constants.types import TensorSample
from tfm.data.base import BaseGenerator


class BlocksWorldGenerator(BaseGenerator):
    # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
    """
    This class generates the dataset for the Blocks World problem. The Blocks
    World problem consists of having a set of blocks in a table, and the goal
    is to move the blocks to a desired configuration. A configuration can be
    having all the blocks in a single stack, or having them in a specific
    order. Each con block can be moved to the top of another block, or to the
    table.

    The blocks are moved by cranes, which can only move one block at a
    time. The cranes can only move blocks that are on the top of a stack, or
    that are on the table. The cranes can only move blocks to the top of a
    stack, or to the table.

    A state is represented by a tensor of shape (n,), where n is the number
    of blocks. Each element of the tensor is an integer that represents the
    object that is below the block. The table is identified by the number -1.
    Each block is identified by a number between 0 and n-1. The cranes are
    from the number -2 to -2 - m + 1, where m is the number of cranes.

    For example, if we have 4 blocks and 2 cranes the identifiers are:

    - Table: -1
    - Cranes: -2, -3
    - Blocks: 0, 1, 2, 3

    Following the example, if we have the following configuration:

    - Block 0 is on the table
    - Block 1 is on the crane -2
    - Block 2 is on the table
    - Block 3 is on the block 2

    The state is represented by the tensor:

    >>> torch.tensor([-1, -2, -1, 2])

    Regarding actions, there is `1 + n + m` actions per block. It means that we
    can move each block to the table, to the top of another block, or to a
    crane. It's possible that some of the actions are not valid. For
    example, if we try to move a block to the top of another block that is
    on a crane. In that case, the action is not valid. So, the actions are:

    - Move to the table: index 0
    - Move to the block i for each block i: indices 1, 2, ..., n
    - Move to the crane j for each crane j: indices n + 1, n + 2, ..., n + m

    Knowing that, each Sample is a tuple of two elements. The first element
    is the state, and the second element is a sequence of possible actions.
    This sequence has the following length:

    >>> (1 + n + m) * n

    Where n is the number of blocks and m is the number of cranes.

    So for examlpe, for a state with 2 blocks and 1 crane, where the first
    block is on the table and the second block is on the first block, the
    sample is:

    >>> (
    ...     torch.tensor([-1, 0]),
    ...     (
    ...         # We can't move the first block to the table because it's already
    ...         # there
    ...         None,
    ...         # We can't move the first block to be on itself
    ...         None,
    ...         # We can't move the first block to the second block because it has
    ...         # another block on top of it
    ...         None,
    ...         # We can't move the first block to crane because it has another
    ...         # block on top of it
    ...         None,
    ...         # We can move the second block to the table
    ...         torch.tensor([-1, -1]),
    ...         # We can move the second block to the first block because it's already
    ...         # there
    ...         None,
    ...         # We can't move the second block to the to be on itself
    ...         None,
    ...         # We can move the second block to crane
    ...         torch.tensor([-1, -2]),
    ...     )
    ... )

    Parameters
    ----------
    sequences : int
        Number of sequences to generate.
    sequence_length : int
        Length of the sequences to generate.
    blocks: int
        The number of blocks in the problem.
    cranes: int
        The number of cranes in the problem.
    """
    def __init__(
        self,
        sequences: int,
        sequence_length: int,
        *, /,
        blocks: int,
        cranes: int,
        shuffle: bool = True,
    ):
        super().__init__(sequences, sequence_length, blocks * cranes + 1, shuffle=shuffle)
        self.blocks = blocks
        self.cranes = cranes

        self.block_size = 32
        self.crane_height = 64
        self.crane_width = 4
        self.margin_size = 8

        color_increment = 255 // blocks
        self.colors = [
            i * color_increment for i in range(blocks)
        ]

    def init_state(self) -> Tensor:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Returns the initial state of the problem. The initial state
        represents that each block is on the table.

        Examples
        --------
        >>> generator = BlocksWorldGenerator(1, 2, blocks=4, cranes=2)
        >>> generator.init_state()
        tensor([-1, -1, -1, -1])

        Returns
        -------
        State
            The initial state of the problem.
        """
        return torch.full((self.blocks,), -1)

    def select(self, sample: TensorSample) -> Tensor:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Given a LightsSample, it selects a Tensor from the sequence of Tensor of
        the LightsSample.

        Parameters
        ----------
        sample : Sample

        Examples
        --------
        >>> generator = BlocksWorldGenerator(1, 2, blocks=2, cranes=1)
        >>> sample = (
        ...     torch.tensor([-1, 0]),
        ...     (
        ...         None,
        ...         None,
        ...         None,
        ...         None,
        ...         torch.tensor([-1, -1]),
        ...         None,
        ...         None,
        ...         torch.tensor([-1, -2]),
        ...     )
        ... )
        >>> generator.select(sample)
        torch.tensor([-1, -2])

        Returns
        -------
        State
            The selected Tensor.
        """
        non_none = [s for s in sample[1] if s is not None]
        return non_none[np.random.randint(len(non_none))]

    def occupied(self, state: Tensor, item: int) -> int:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Given a state and an item it checks if the item occupied by another. If
        the item is a block, then it checks if the block is under another and
        if the item is a crane, then it checks if the crane is occupied.

        Parameters
        ----------
        state : State
            The state of the problem.
        item : int
            The item to check if it's occupied.

        Examples
        --------
        >>> generator = BlocksWorldGenerator(1, 2, blocks=2, cranes=1)
        >>> state = torch.tensor([-1, -1])
        >>> generator.occupied(state, 0)
        False

        >>> state = torch.tensor([-1, -2])
        >>> generator.occupied(state, -2)
        True

        Returns
        -------
        bool
            True if the item is occupied, False otherwise.
        """
        return len((state == item).nonzero().squeeze()) > 0

    def move(self, state: Tensor, action: int) -> Tensor | None:
        # noinspection PyShadowingNames,PyTypeChecker,PyUnresolvedReferences
        """
        Given a state and an action, it returns the new state after applying
        the action. If the action is not valid, it returns None.

        First of all, the action is divided into three parts: block and where
        to move it. Then we check if this block is under another block. If
        it's under another block, then the action is not valid. After this,
        we check if the action is to move it to itself. If it's the case,
        then the action is not valid.

        Then we check if the action is to move it to the table. If it's the
        case, then we move it to the table and return the new state. If the
        action is to move it to another block or to a crane, then we check if
        the object is occupied by another block. If it's the case, then the
        action is not valid.

        If it's not the case, then we move the block to the object and return
        the new state. To get the id of the object we take into account the
        way we have defined the possible actions.

        Am action is a number between 0 and (1 + n + m) * n - 1. This can be
        represented on the next way:

        - Possible action to apply to the block 0
        - Possible action to apply to the block 1
        - ...
        - Possible action to apply to the block n - 1

        The possible actions to apply to the block i are:

        - Move it to the table
        - Move it to the block 0
        - Move it to the block 1
        - ...
        - Move it to the block n - 1
        - Move it to the crane -2
        - Move it to the crane -3
        - ...
        - Move it to the crane -2 - m + 1

        So it can be represented as a matrix of shape (n, 1 + n + m), where
        each row represents the possible actions to apply to the block i.

        >>> # block 0     -> [0, 1, 2, ..., n, n + 1, n + 2, ..., n + m]
        >>> # block 1     -> [0, 1, 2, ..., n, n + 1, n + 2, ..., n + m]
        >>> # ...
        >>> # block n - 1 -> [0, 1, 2, ..., n, n + 1, n + 2, ..., n + m]

        The mapping between the action and the object is:

        >>> # block 0     -> [-1, 0, 1, ..., n - 1, -2, -3, ..., -2 - m + 1]
        >>> # block 1     -> [-1, 0, 1, ..., n - 1, -2, -3, ..., -2 - m + 1]
        >>> # ...
        >>> # block n - 1 -> [-1, 0, 1, ..., n - 1, -2, -3, ..., -2 - m + 1]

        So here is the explanation of how we can get the object id from the
        action:

        >>> # We get the block id and the object id
        >>> block, item = divmod(action, n + m + 1)
        >>> # If the action is to move it to the table
        >>> new_state[block] = -1
        >>> # If the action is to move it to another block
        >>> new_state[block] = item - 1
        >>> # If the action is to move it to a crane
        >>> new_state[block] = item - (1 + n) - 2

        Parameters
        ----------
        state : State
            The state of the problem.
        action : int
            The action to apply to the state.

        Examples
        --------
        >>> generator = BlocksWorldGenerator(1, 2, blocks=2, cranes=1)
        >>> state = torch.tensor([1, -1])
        >>> generator.move(state, 0)
        tensor([-1, -1])

        >>> generator.move(state, 1)
        None

        Returns
        -------
        State
            The new state after applying the action.
        """
        new_state = state.clone()
        block, item = divmod(action, self.blocks + self.cranes + 1)

        # The block is under another block
        if self.occupied(state, block):
            return None

        # The action is move it to itself
        if block == item - 1:
            return None

        # Check if the action is to move it to the table
        if item == 0:
            new_state[block] = -1
            return new_state

        # If the action to move the block to an object that is occupied by
        # another block
        if self.occupied(state, item):
            return None

        # If the action is to move to a crane, then object index is negative
        if item > self.blocks:
            new_state[block] = item - (1 + self.blocks) - 2
        else:
            # The action is move it to another block
            new_state[block] = item - 1

        return new_state

    def image(self, state: Tensor) -> Tensor:
        pass
