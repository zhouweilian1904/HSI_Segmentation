import torch
import torch.nn as nn


def to_uturn_sequence(img_tensor):
    # Assuming img_tensor is your input tensor with shape (batch, channel, height, width)
    batch, channel, height, width = img_tensor.shape

    # Permute the tensor to bring height and width to the end
    # New shape: (batch, channel, height, width) -> (batch, height, width, channel)
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    # Initialize an empty list to hold each row after applying the U-Turn pattern
    uturn_rows = []

    # Loop through each row, reversing the order in every other row
    for row_index in range(height):
        if row_index % 2 == 0:
            # Even rows remain as is
            uturn_rows.append(img_tensor[:, row_index, :, :])
        else:
            # Odd rows get reversed
            uturn_rows.append(img_tensor[:, row_index, :, :].flip(dims=[1]))

    # Stack the rows back together along the height dimension
    # And reshape to (batch, height * width, channel)
    sequence = torch.stack(uturn_rows, dim=1).reshape(batch, -1, channel)

    return sequence


# Example usage:
# img_tensor is your input tensor with shape (batch, channel, height, width)
# Replace img_tensor with your actual tensor variable
# sequence = to_uturn_sequence(img_tensor)
# Now, sequence has the shape (batch, length, dimension)


import torch


def snake_flatten(img_tensor):
    # img_tensor is expected to be of shape (batch, channel, width, height)
    batch_size, channels, width, height = img_tensor.size()

    # Permute the tensor to bring the channels to the last dimension
    img_tensor = img_tensor.permute(0, 3, 2, 1)  # New shape: (batch, height, width, channel)

    # Clone the tensor to avoid in-place modifications
    img_tensor = img_tensor.clone()

    # Applying the snake pattern by reversing every other row
    for i in range(height):
        if i % 2 != 0:  # Reverse the order of pixels in every odd row
            img_tensor[:, i] = img_tensor[:, i, :].flip(dims=[1])

    # Reshape to flatten the height and width into a single dimension, maintaining the batch and channel dimensions
    return img_tensor.reshape(batch_size, -1, channels)


def snake_unflatten(sequence, original_shape):
    # original_shape is expected to be (batch, channel, width, height)
    batch_size, channels, width, height = original_shape

    # Calculate height and width from the original shape
    img_tensor = sequence.view(batch_size, height, width, channels)

    # Clone the tensor to avoid in-place modifications
    img_tensor = img_tensor.clone()

    # Reverse the snaking pattern by flipping every alternate row back
    for i in range(height):
        if i % 2 != 0:  # Check for odd rows that were flipped
            img_tensor[:, i] = img_tensor[:, i, :].flip(dims=[1])

    # Permute back to (batch, channel, width, height)
    img_tensor = img_tensor.permute(0, 3, 2, 1)

    return img_tensor


