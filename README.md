
 🧬 MaxSigLayer: A Learnable Layer for Semantic Cell Segmentation

Implementation of the novel MaxSigLayer from
"MaxSigNet: Light Learnable Layer for Semantic Cell Segmentation"
by Reza Yazdi and Hassan Khotanlou (2024)

This repository offers a clean, modular, and efficient PyTorch implementation of MaxSigLayer — a lightweight, learnable layer designed to enhance fine-grained feature extraction in semantic segmentation tasks.

✅ Plug-and-play compatible with:

> - U-Net and its variants

> - Biomedical segmentation pipelines

> - Preprocessing blocks in deep learning workflows

---

📄 Paper Summary & Core Concept

MaxSigLayer is introduced as a lightweight, learnable filtering layer, drawing inspiration from traditional Sobel filters and Local Binary Patterns (LBP), but extended with the flexibility of deep learning.

Unlike fixed filters, MaxSigLayer learns optimal weights to compare each pixel’s neighborhood with a trainable kernel. It then combines:

The maximum of sigmoid-activated neighbor and kernel weights

The median and mean of these localized comparisons

The original center pixel value

This combination produces a smooth, non-linear feature map that enhances foreground-background contrast, making it particularly effective in cell segmentation, where boundaries are often ambiguous and cells overlap.

The broader architecture, MaxSigNet, stacks multiple MaxSigLayers along with dilated convolutions and edge-awareness modules. It demonstrates state-of-the-art performance on diverse cell segmentation datasets (CTC, BBBC039, LIVECell), and generalizes well to other modalities like CT, MRI, and ultrasound.

✅ Key Innovations

> - Learnable Local Statistics: Replaces hardcoded LBP and Sobel thresholds with trainable parameters.

> - Smooth Non-Linearity: Uses sigmoid activation to ensure stable gradients and bounded outputs.

> - Lightweight & Efficient: Delivers superior performance compared to U-Net and transformers, with significantly fewer parameters and lower computational cost.

> - Cross-Modal Generalization: Successfully validated across microscopy, CT, MRI, and ultrasound imaging.

---

 💻 Code Implementation Breakdown
```python

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
```
- `ch_in`: Input channels (e.g., 1 for grayscale, 3 for RGB)
- `ch_out`: Output channels — allows channel expansion/reduction
- `kernel_size`: Spatial size of local neighborhood (e.g., 5, 7, 9)
- `stride`: Controls output spatial resolution (default 1 = same size)
- Internally computes `padding` to preserve spatial dimensions
- Initializes:
  - `weight`: Learnable kernel weights (applied per-patch location)
  - `bias`: Learnable bias per patch location
  - `weight_center`, `weight_median`: Learnable scalars to balance contributions

 🔄 Forward Pass (`forward`)
```python
def forward(self, x):
```
1. Applies `sigmoid` to input for non-linear normalization.
2. Uses `F.unfold` to extract local patches across the image (efficient sliding window).
3. Compares each patch element-wise with learned weights via `torch.maximum`.
4. Computes:
   - `max_stat`: Sum of max values across patch
   - `median_stat`: Median of max values
   - `mean_stat`: Mean of max values
   - `center_values`: Original center pixel from patch
5. Combines them with learnable weights:
   ```
   out = (
       self.weight_center * center_values +
       self.weight_median * median_stat -
       max_stat -
       mean_stat
   )
   ```
6. Expands to `ch_out` output channels via summation across input channels.

---

 🚀 How to Use

 1. Install Dependencies
```bash
pip install torch torchvision
```

 2. Import and Instantiate
```python
from maxsiglayer import MaxSigLayer

 Create layer: 3 input channels, 64 output channels, 5x5 kernel
layer = MaxSigLayer(ch_in=3, ch_out=64, kernel_size=5)

 Forward pass
x = torch.randn(1, 3, 256, 256)   Example input
y = layer(x)   Output: (1, 64, 256, 256)
print(y.shape)
```

 3. Plug into Your Model
```python
import torch.nn as nn

class MySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxsig = MaxSigLayer(3, 64, 7)
        self.conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.maxsig(x)
        x = self.conv(x)
        return torch.sigmoid(x)
```

---

 📚 Citation

If you use this code or are inspired by the MaxSigLayer concept in your research, please cite the original paper:

```bibtex
@article{yazdi2024maxsignet,
  title = {MaxSigNet: Light learnable layer for semantic cell segmentation},
  journal = {Biomedical Signal Processing and Control},
  volume = {95},
  pages = {106464},
  year = {2024},
  issn = {1746-8094},
  doi = {https://doi.org/10.1016/j.bspc.2024.106464},
  author = {Reza Yazdi and Hassan Khotanlou},
}
```

Or 
 
> Yazdi, R., & Khotanlou, H. (2024). MaxSigNet: Light learnable layer for semantic cell
segmentation. Biomed. Signal Process. Control., 95, 106464. https://doi.org/10.1016/j.bspc.2024.106464

---
