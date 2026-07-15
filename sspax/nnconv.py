"""cuDNN-free 2D convolution (im2col + matmul) as a flax drop-in.

This box's cuDNN mismatches jaxlib (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH),
so lax conv (which dispatches to cuDNN on GPU) fails. im2col + matmul uses only
pad / strided-slice / reshape / matmul -> cuBLAS + elementwise, all fine on the
accelerator. The nets here are tiny, so the extra memory of explicit patches is
negligible and it runs on GPU at full speed.
"""
import jax.numpy as jnp
import flax.linen as nn


def conv2d(x, w, b, stride=1, padding="SAME"):
    """x (B,H,W,Cin), w (kh,kw,Cin,Cout), b (Cout,) -> (B,oH,oW,Cout)."""
    kh, kw, Cin, Cout = w.shape
    B, H, W, _ = x.shape
    if padding == "SAME":
        oH = -(-H // stride)
        oW = -(-W // stride)
        ph = max(0, (oH - 1) * stride + kh - H)
        pw = max(0, (oW - 1) * stride + kw - W)
        x = jnp.pad(x, ((0, 0), (ph // 2, ph - ph // 2),
                        (pw // 2, pw - pw // 2), (0, 0)))
    Hp, Wp = x.shape[1], x.shape[2]
    oH = (Hp - kh) // stride + 1
    oW = (Wp - kw) // stride + 1
    cols = []
    for i in range(kh):
        for j in range(kw):
            cols.append(x[:, i:i + (oH - 1) * stride + 1:stride,
                          j:j + (oW - 1) * stride + 1:stride, :])
    col = jnp.stack(cols, -1).reshape(B, oH, oW, Cin * kh * kw)
    wm = w.transpose(2, 0, 1, 3).reshape(Cin * kh * kw, Cout)
    return col @ wm + b


def crelu(x):
    """Concatenated ReLU: [relu(x), relu(-x)] on the channel axis. Doubles the
    activation width from the SAME filters (both phases kept), so a layer needs
    ~half the filters for comparable capacity — param-efficient, and on-fabric
    it is just a sign flip + relu (no extra MACs). Halves the next layer's input
    channels' weight count too."""
    return jnp.concatenate([nn.relu(x), nn.relu(-x)], axis=-1)


class Conv(nn.Module):
    features: int
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x):
        kh, kw = self.kernel_size
        Cin = x.shape[-1]
        w = self.param("kernel", nn.initializers.lecun_normal(),
                       (kh, kw, Cin, self.features))
        b = self.param("bias", nn.initializers.zeros, (self.features,))
        s = self.strides[0] if isinstance(self.strides, (tuple, list)) \
            else self.strides
        return conv2d(x, w, b, s, self.padding)
