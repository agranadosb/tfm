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
    can move each block to the top of another block, to the table or to a
    crane. It's possible that some of the actions are not valid. For
    example, if we try to move a block to the top of another block that is
    on a crane. In that case, the action is not valid. So, the actions are:

    - Move to the table: 0
    - Move to the block i for each block i: 1, 2, ..., n
    - Move to the crane j for each crane j: -1 - 1, -1 - 2, ..., -1 - m

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

    def move(self, state: Tensor, action: int) -> Tensor | None:
        pass

    def image(self, state: Tensor) -> Tensor:
        pass
