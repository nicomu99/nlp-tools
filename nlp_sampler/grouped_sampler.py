from typing import List, Iterator
import random

from torch.utils.data import Sampler

class GroupedSampler(Sampler):
    def __init__(self, input_ids: List[List[int]], batch_size: int):
        """
        Initialize the grouped sampler. Create a list of sequence lengths.

        :param input_ids: Torch tensor of input features to be sorted.
        :param batch_size: The training batch size.
        """
        super().__init__()

        self.sequence_count = len(input_ids)
        self.batch_size = batch_size
        self.group_batch_size = self.batch_size * 100

        # Pair each sequence index with its tokenized sequence length
        self.seq_length_index = [(index, len(sequence)) for index, sequence in enumerate(input_ids)]


    def __iter__(self) -> Iterator[int]:
        """
        Shuffles the data, creates groups of some pre-defined size and sorts each group by the length of the inputs in
        descending order.

        :return: An iterator over the input ids.
        """
        # Shuffle the list
        random.shuffle(self.seq_length_index)

        # Generate groups of size BATCH_SIZE * 100
        sorted_batch_list = []
        for i in range(0, len(self.seq_length_index), self.group_batch_size):
            next_batch = self.seq_length_index[i:i + self.group_batch_size]
            # Sort each group by its sequence length
            next_batch = sorted(next_batch, key=lambda x: x[1], reverse=True)
            sorted_batch_list.extend([index for index, _ in next_batch])

        # Return a list of tuples sorted by ascending sequence length
        return iter(sorted_batch_list)


    def __len__(self) -> int:
        return self.sequence_count