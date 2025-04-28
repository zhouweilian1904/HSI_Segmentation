import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class OptimizedRNN2DCell(nn.Module):
    def __init__(self, input_size, hidden_size, device='cuda'):
        super(OptimizedRNN2DCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # Weights for input, hidden states from above and left
        self.W_ih = nn.Linear(input_size, hidden_size).to(self.device)
        self.W_hh_top = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.W_hh_left = nn.Linear(hidden_size, hidden_size).to(self.device)

    def forward(self, x, hidden_top, hidden_left):
        # Combine the hidden states from top and left, and the current input
        hidden_next = F.relu(self.W_ih(x) + self.W_hh_top(hidden_top) + self.W_hh_left(hidden_left))
        return hidden_next


class OptimizedRNN2D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, height, width, device='cuda'):
        super(OptimizedRNN2D, self).__init__()
        self.hidden_size = hidden_size
        self.height = height
        self.width = width
        self.device = device
        self.rnn_cell = OptimizedRNN2DCell(input_size, hidden_size, device=device)

        # Point-wise convolution (1x1 convolution) to map hidden states to output size
        self.pointwise_conv = nn.Conv2d(in_channels=hidden_size, out_channels=output_size, kernel_size=1).to(
            self.device)

    def forward(self, x):
        batch_size, height, width, input_size = x.size()

        # Initialize hidden states for the entire grid on the correct device
        hidden_states = torch.zeros(batch_size, height, width, self.hidden_size, device=self.device)

        # Process each row in the image at once
        for i in range(height):
            # Collect the previous hidden states (top and left) for the entire row
            hidden_top = hidden_states[:, i - 1, :, :] if i > 0 else torch.zeros(batch_size, width, self.hidden_size,
                                                                                 device=self.device)
            hidden_left = torch.cat(
                [torch.zeros(batch_size, 1, self.hidden_size, device=self.device), hidden_states[:, i, :-1, :]], dim=1)

            # Forward pass for the entire row in parallel
            hidden_states[:, i, :, :] = self.rnn_cell(x[:, i, :, :], hidden_top, hidden_left)

        # Reshape hidden states for convolution: (batch_size, hidden_size, height, width)
        hidden_states = hidden_states.permute(0, 3, 1, 2)  # Change to (batch_size, hidden_size, height, width)

        # Apply point-wise convolution
        outputs = self.pointwise_conv(hidden_states)  # Shape: (batch_size, output_size, height, width)

        return F.relu(outputs), F.relu(hidden_states)


class FourDirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, height, width, device='cuda'):
        super(FourDirectionalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.height = height
        self.width = width
        self.device = device

        # Optimized RNN2D for four different directions
        self.rnn_top_left = OptimizedRNN2D(input_size, hidden_size, output_size, height, width, device=device)
        self.rnn_top_right = OptimizedRNN2D(input_size, hidden_size, output_size, height, width, device=device)
        self.rnn_bottom_left = OptimizedRNN2D(input_size, hidden_size, output_size, height, width, device=device)
        self.rnn_bottom_right = OptimizedRNN2D(input_size, hidden_size, output_size, height, width, device=device)

        # Learnable weights for each direction
        self.direction_weights = nn.Parameter(torch.ones(4))  # Initializing weights as 1 for each direction

    def forward(self, x):
        batch_size, height, width, input_size = x.size()

        # Top-left to bottom-right
        out_top_left, _ = self.rnn_top_left(x)

        # Top-right to bottom-left: reverse columns (flip along width axis)
        x_flipped_lr = torch.flip(x, dims=[2])
        out_top_right, _ = self.rnn_top_right(x_flipped_lr)
        out_top_right = torch.flip(out_top_right, dims=[2])  # Re-flip to restore original orientation

        # Bottom-left to top-right: reverse rows (flip along height axis)
        x_flipped_ud = torch.flip(x, dims=[1])
        out_bottom_left, _ = self.rnn_bottom_left(x_flipped_ud)
        out_bottom_left = torch.flip(out_bottom_left, dims=[1])  # Re-flip to restore original orientation

        # Bottom-right to top-left: reverse both rows and columns (flip along both height and width)
        x_flipped_ud_lr = torch.flip(x, dims=[1, 2])
        out_bottom_right, _ = self.rnn_bottom_right(x_flipped_ud_lr)
        out_bottom_right = torch.flip(out_bottom_right, dims=[1, 2])  # Re-flip to restore original orientation

        # Combine the four directional outputs using weighted averaging
        combined_output = (
                                  self.direction_weights[0] * out_top_left +
                                  self.direction_weights[1] * out_top_right +
                                  self.direction_weights[2] * out_bottom_left +
                                  self.direction_weights[3] * out_bottom_right
                          ) / self.direction_weights.sum()

        return F.relu(combined_output.permute(0, 2, 3, 1))


class hsi_branch(nn.Module):
    def __init__(self, in_channels, hidden_channels, output_channels, patch_size, num_layers=2, type='down'):
        super(hsi_branch, self).__init__()
        self.encoder = FourDirectionalRNN(input_size=in_channels, hidden_size=hidden_channels,
                                          output_size=output_channels, height=patch_size, width=patch_size)
        self.layers = nn.ModuleList()
        # Append the encoder layers
        for _ in range(num_layers):
            self.layers.append(
                self.encoder
            )

        self.type = type
        self.down = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1)
        self.up = nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3, stride=1)

    def forward(self, x):
        for four_rnn in self.layers:
            out = four_rnn(x)
            out = out + F.relu(x)
            out = out.permute(0, 3, 1, 2)
            if self.type == 'down':
                out = self.down(out)
            else:
                out = self.up(out)
        return out.permute(0, 2, 3, 1)


class x_data_branch(nn.Module):
    def __init__(self, in_channels, hidden_channels, output_channels, patch_size, num_layers=2, type='down'):
        super(x_data_branch, self).__init__()
        self.encoder = FourDirectionalRNN(input_size=in_channels, hidden_size=hidden_channels,
                                          output_size=output_channels, height=patch_size, width=patch_size)
        self.layers = nn.ModuleList()
        # Append the encoder layers
        for _ in range(num_layers):
            self.layers.append(
                self.encoder
            )

        self.type = type
        self.down = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1)
        self.up = nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3, stride=1)

    def forward(self, x):
        for four_rnn in self.layers:
            out = four_rnn(x)
            out = out + F.relu(x)
            out = out.permute(0, 3, 1, 2)
            if self.type == 'down':
                out = self.down(out)
            else:
                out = self.up(out)
        return out.permute(0, 2, 3, 1)


class Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, output_channels, patch_size, num_class, num_layers=1):
        super(Net, self).__init__()

        self.hsi_embd = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.hsi_encoder_down = hsi_branch(hidden_channels, hidden_channels, output_channels, patch_size,
                                      num_layers=num_layers, type='down')
        self.hsi_encoder_up = hsi_branch(hidden_channels, hidden_channels, output_channels, patch_size,
                                           num_layers=num_layers, type='up')

        self.x_data_embd = nn.Conv2d(1, hidden_channels, kernel_size=1)

        self.x_data_encoder_down = x_data_branch(hidden_channels, hidden_channels, output_channels, patch_size,
                                            num_layers=num_layers, type='down')
        self.x_data_encoder_up = x_data_branch(hidden_channels, hidden_channels, output_channels, patch_size,
                                                 num_layers=num_layers, type='up')

        self.decoder_seg = nn.Conv2d(output_channels, num_class, kernel_size=1)
        self.aux_rec = nn.Conv2d(output_channels, 1, kernel_size=1)
        self.decoder_cls = nn.Sequential(
            nn.Linear(output_channels * 2, output_channels),
            nn.ReLU(inplace=True),
            nn.Linear(output_channels, num_class)
        )

    def forward(self, hsi, x_data):
        hsi = hsi.squeeze(1)
        hsi_0 = self.hsi_embd(hsi)
        hsi_0 = rearrange(hsi_0, 'b c h w -> b h w c')

        hsi_1 = self.hsi_encoder_down(hsi_0)
        hsi_2 = self.hsi_encoder_down(hsi_1)
        hsi_3 = self.hsi_encoder_down(hsi_2)
        hsi_4 = self.hsi_encoder_down(hsi_3)

        hsi = self.hsi_encoder_up(hsi_4)
        hsi = self.hsi_encoder_up(hsi)
        hsi = self.hsi_encoder_up(hsi)
        hsi = self.hsi_encoder_up(hsi)

        x_data = x_data.squeeze(1)
        x_data_0 = self.x_data_embd(x_data)
        x_data_0 = rearrange(x_data_0, 'b c h w -> b h w c')

        x_data_1 = self.x_data_encoder_down(x_data_0)
        x_data_2 = self.x_data_encoder_down(x_data_1)
        x_data_3 = self.x_data_encoder_down(x_data_2)
        x_data_4 = self.x_data_encoder_down(x_data_3)

        x_data = self.x_data_encoder_up(x_data_4)
        x_data = self.x_data_encoder_up(x_data)
        x_data = self.x_data_encoder_up(x_data)
        x_data = self.x_data_encoder_up(x_data)

        com = torch.cat([hsi_4, x_data_4], dim=-1)
        out_seg = self.decoder_seg(rearrange(F.relu(hsi), 'b h w d -> b d h w'))
        aux_seg = self.decoder_seg(rearrange(F.relu(x_data), 'b h w d -> b d h w'))
        aux_rec = self.aux_rec(rearrange(F.relu(x_data), 'b h w d -> b d h w'))

        x_global = com.mean(dim=[1, 2])
        out_cls = self.decoder_cls(F.relu(x_global))

        return out_seg, out_cls, aux_seg, aux_rec
