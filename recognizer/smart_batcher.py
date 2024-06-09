import os
from typing import Any, Sequence

import imagesize
import numpy as np
from PIL import Image

from . import constants


class SmartBatcher:
    """
    A class to perform smart batching for the Recognizer.
    """

    # A dictionary of functions to get the width of an image.
    SORT_FUNCTIONS = {
        str: lambda image_index_pair: imagesize.get(image_index_pair[0])[0],
        Image.Image: lambda image_index_pair: image_index_pair[0].width,
        np.ndarray: lambda image_index_pair: image_index_pair[0].shape[1],
    }

    @staticmethod
    def is_supported(input_data: Any, verbose: bool = False) -> bool:
        """Checks if smart batching is supported for the given input data"""
        if isinstance(input_data, constants.SMART_BATCHER_SUPPORTED_TYPES):
            if verbose:
                print("Smart batching enabled. Images will be sorted by width before inference.")
            return True
        if verbose:
            print(f"Smart batching is not supported for {type(input_data)}. Proceeding without smart batching.")
        return False

    @staticmethod
    def split_sequence_to_batches(inputs: Sequence[Any], batch_size: int) -> list[Sequence[Any]]:
        """Split any sequence into batches of the given batch_size"""
        batches = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
        return batches

    @staticmethod
    def sort_items(
        inputs: Sequence[str | Image.Image | np.ndarray]
    ) -> tuple[list[str | Image.Image | np.ndarray], list[int]]:
        """
        Sorts a sequence of inputs by width and returns the sorted inputs and the original order of the inputs.

        Arguments
        ---------
            - inputs (Sequence[str | Image.Image | np.ndarray]): A sequence of image paths, PIL images or numpy arrays.

        Returns
        -------
            tuple[list[Image.Image | np.ndarray], list[int]]: A tuple of the sorted inputs and the original order of the inputs.

        Example
        -------
            >>> inputs = [image_1, image_2, image_3]
            >>> sorted_inputs, sort_orders = SmartBatcher.sort_items(inputs)
            >>> sorted_inputs
            [image_3, image_1, image_2]
            >>> sort_orders
            [2, 0, 1]
        """

        # Check if the inputs are the same type.
        all_input_types = set(type(input) for input in inputs)
        if len(all_input_types) > 1:
            raise TypeError(f"All inputs must be of the same type. Got: {all_input_types}")

        # Check if the inputs are supported.
        input_sample = inputs[0]
        is_supported = SmartBatcher.is_supported(input_sample)
        if not is_supported:
            return inputs, list(range(len(inputs)))

        # If the input is a list of paths, then check if all paths exist.
        if isinstance(input_sample, str):
            for path in inputs:
                if not os.path.exists(path):
                    raise FileNotFoundError(f'Received invalid path: "{path}"')

        # Create a zip object of (image, index) pairs.
        image_index_pair_zip = tuple(zip(inputs, range(len(inputs))))

        # Get the function to get the width of an image based on the input type.
        input_type = type(input_sample)
        sort_function = SmartBatcher.SORT_FUNCTIONS[input_type]

        # Sort the pairs by the width of the images. The reverse=True means that the images will be sorted in descending order.
        sorted_image_index_pair_zip = sorted(image_index_pair_zip, key=sort_function, reverse=True)

        # Unzip the sorted_image_index_pair_zip to get the sorted_image_list and original_index_list.
        sorted_image_list, original_index_list = map(list, zip(*sorted_image_index_pair_zip))

        return sorted_image_list, original_index_list

    @staticmethod
    def reorder_items(inputs: Sequence[Any], sort_orders: Sequence[int]) -> list[Any]:
        """
        Reorders the inputs by the given sort_orders.

        Arguments
        ---------
            - inputs (Sequence[Any]): A sequence of inputs.
            - sort_orders (Sequence[int]): A sequence of indices to sort the inputs.

        Returns
        -------
            list[Any]: The reordered inputs.

        Example
        -------
            >>> inputs = ["apple", "banana", "orange"]
            >>> sort_orders = [2, 0, 1]
            >>> reordered_inputs = SmartBatcher.reorder_items(inputs, sort_orders)
            >>> reordered_inputs
            ["banana", "orange", "apple"]
        """
        sorted_inputs = sorted(zip(inputs, sort_orders), key=lambda pair: pair[1])
        sorted_inputs = [pair[0] for pair in sorted_inputs]
        return sorted_inputs
