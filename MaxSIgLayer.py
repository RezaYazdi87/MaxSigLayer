import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxSigLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1):
        """
        A custom learnable layer combining local statistics (max, median, mean) with sigmoid activations.
        
        Args:
            ch_in (int): Number of input channels.
            ch_out (int): Number of output channels.
            kernel_size (int): Size of the kernel (assumed square).
            stride (int): Stride for unfolding operation.
        """
        super(MaxSigLayer, self).__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.kernel_area = kernel_size ** 2
        self.center_index = self.kernel_area // 2

        # Learnable weights and bias
        self.weight = nn.Parameter(torch.rand(1, 1, self.kernel_area, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1, self.kernel_area, 1, 1))

        # Additional learnable scalars for combining center pixel, median, etc.
        self.weight_center = nn.Parameter(torch.tensor(1.0))
        self.weight_median = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        Forward pass for MaxSigLayer.

        Args:
            x (Tensor): Input tensor of shape (B, C_in, H, W)

        Returns:
            Tensor: Output tensor of shape (B, C_out, H', W')
        """
        B, C_in, H, W = x.shape

        # Apply sigmoid and unfold to extract local patches
        x_sigmoid = torch.sigmoid(x)
        x_unfold = F.unfold(x_sigmoid, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        x_unfold = x_unfold.view(B, C_in, self.kernel_area, H, W)  # (B, C_in, K*K, H, W)

        # Sigmoid of learnable weights and bias
        w = torch.sigmoid(self.weight)  # (1, 1, K*K, 1, 1)
        b = torch.sigmoid(self.bias)    # (1, 1, K*K, 1, 1)

        # Compute D = max(w, x_unfold) + b
        D = torch.maximum(w, x_unfold) + b  # (B, C_in, K*K, H, W)

        # Compute statistics across the patch dimension
        max_stat = torch.sum(D, dim=2)           # (B, C_in, H, W)
        median_stat, _ = torch.median(D, dim=2)  # (B, C_in, H, W)
        mean_stat = torch.mean(D, dim=2)         # (B, C_in, H, W)

        # Extract center pixel values
        center_values = x_unfold[:, :, self.center_index, :, :]  # (B, C_in, H, W)

        # Combine using learnable weights
        out = (
            self.weight_center * center_values +
            self.weight_median * median_stat -
            max_stat -
            mean_stat
        )  # (B, C_in, H, W)

        # Expand to output channels
        out = out.unsqueeze(1).expand(-1, self.ch_out, -1, -1, -1)  # (B, C_out, C_in, H, W)
        out = torch.sum(out, dim=2)  # (B, C_out, H, W)

        return out