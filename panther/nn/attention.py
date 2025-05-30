import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from panther.nn.pawXimpl import create_projection_matrix, rmha_forward, sinSRPE


def verify_rmha_inputs(
    embed_dim,
    num_heads,
    dropout,
    bias,
    kernel_fn,
    iscausal,
):  # verfies all the performers inputs
    """
    Validates the input parameters for a multi-head attention mechanism.

    Parameters:
        embed_dim (int): The embedding dimension. Must be positive and divisible by num_heads.
        num_heads (int): The number of attention heads. Must be positive.
        dropout (float): Dropout probability. Must be in the range [0, 1).
        bias (bool): Whether to use bias in the attention mechanism. Must be a boolean.
        kernel_fn (str): The kernel function to use. Must be either 'softmax' or 'relu'.
        iscausal (bool): Whether the attention is causal. Must be a boolean.

    Raises:
        ValueError: If any of the input parameters do not meet the specified requirements.
    """
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if embed_dim <= 0:
        raise ValueError("embed_dim must be positive")
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("dropout must be in the range [0, 1)")
    if not isinstance(bias, bool):
        raise ValueError("bias must be a boolean value")
    if kernel_fn not in ["softmax", "relu"]:
        raise ValueError("kernel_fn must be either 'softmax' or 'relu'")
    if not isinstance(iscausal, bool):
        raise ValueError("iscausal must be a boolean value")


class RandMultiHeadAttention(nn.Module):
    """
    RandMultiHeadAttention implements a random feature-based multi-head attention layer.

    This module projects queries, keys, and values using learned linear transformations,
    optionally adds biases, and applies a random feature projection to approximate
    attention mechanisms such as softmax attention. The layer supports multiple heads,
    dropout, and optional causal masking.

    Args:
        embed_dim (int): Total dimension of the model (input and output features).
        num_heads (int): Number of attention heads.
        num_random_features (int): Number of random features for the projection matrix.
        dropout (float, optional): Dropout probability on attention weights. Default: 0.0.
        bias (bool, optional): If True, adds bias to the linear projections. Default: True.
        kernel_fn (str, optional): Kernel function to use for random feature mapping (e.g., "softmax"). Default: "softmax".
        iscausal (bool, optional): If True, applies causal masking for autoregressive tasks. Default: False.
        device (torch.device, optional): Device for parameters and buffers.
        dtype (torch.dtype, optional): Data type for parameters and buffers.

    Attributes:
        Wq (torch.Tensor): Query projection weight matrix.
        Wk (torch.Tensor): Key projection weight matrix.
        Wv (torch.Tensor): Value projection weight matrix.
        W0 (torch.Tensor): Output projection weight matrix.
        bq (torch.Tensor or None): Query projection bias.
        bk (torch.Tensor or None): Key projection bias.
        bv (torch.Tensor or None): Value projection bias.
        b0 (torch.Tensor or None): Output projection bias.
        projection_matrix (torch.Tensor): Random feature projection matrix.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        bias (bool): Whether biases are used.
        kernel_fn (str): Kernel function for random features.
        causal (bool): Whether causal masking is applied.

    Methods:
        forward(query, key, value, attention_mask=None):
            Computes the multi-head attention output using random feature approximation.
            Args:
                query (torch.Tensor): Input query tensor of shape (batch, seq_len, embed_dim).
                key (torch.Tensor): Input key tensor of shape (batch, seq_len, embed_dim).
                value (torch.Tensor): Input value tensor of shape (batch, seq_len, embed_dim).
                attention_mask (torch.Tensor, optional): Optional mask tensor.
            Returns:
                Tuple[torch.Tensor, None]: Output tensor and None (for compatibility).
    """

    projection_matrix: torch.Tensor
    Wq: torch.Tensor
    Wk: torch.Tensor
    Wv: torch.Tensor
    W0: torch.Tensor
    bq: torch.Tensor | None
    bk: torch.Tensor | None
    bv: torch.Tensor | None
    b0: torch.Tensor | None
    embed_dim: int
    num_heads: int
    dropout: float
    bias: bool
    kernel_fn: str
    causal: bool

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_random_features: int,
        dropout: float = 0.0,
        bias: bool = True,
        kernel_fn: str = "softmax",
        iscausal: bool = False,
        SRPE: sinSRPE | None = None,
        device=None,
        dtype=None,
    ) -> None:
        verify_rmha_inputs(embed_dim, num_heads, dropout, bias, kernel_fn, iscausal)
        super().__init__()
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.dropout: float = dropout
        self.bias: bool = bias
        self.kernel_fn: str = kernel_fn
        self.causal: bool = iscausal
        factory_kwargs = {"dtype": dtype, "device": device}

        self.Wq = nn.Parameter(torch.empty(embed_dim, embed_dim, **factory_kwargs))
        self.Wk = nn.Parameter(torch.empty(embed_dim, embed_dim, **factory_kwargs))
        self.Wv = nn.Parameter(torch.empty(embed_dim, embed_dim, **factory_kwargs))
        self.W0 = nn.Parameter(torch.empty(embed_dim, embed_dim, **factory_kwargs))

        if bias:
            self.bq = nn.Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.bk = nn.Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.bv = nn.Parameter(torch.empty(embed_dim, **factory_kwargs))
            self.b0 = nn.Parameter(torch.empty(embed_dim, **factory_kwargs))
        else:
            self.bq = self.bk = self.bv = self.b0 = None
        self.srpe = SRPE
        self.register_buffer(
            "projection_matrix",
            create_projection_matrix(
                num_random_features,
                embed_dim // num_heads
                if self.srpe is None
                else self.srpe.num_realizations,
                **factory_kwargs,
            ),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initializes or resets the parameters of the attention layer.

        This method applies Xavier uniform initialization to the weight matrices (Wq, Wk, Wv, W0).
        If biases are used, it sets all bias tensors (bq, bk, bv, b0) to zero, provided they are not None.
        """
        xavier_uniform_(self.Wq)
        xavier_uniform_(self.Wk)
        xavier_uniform_(self.Wv)
        xavier_uniform_(self.W0)
        if self.bias:
            if self.bq is not None:
                constant_(self.bq, 0.0)
            if self.bk is not None:
                constant_(self.bk, 0.0)
            if self.bv is not None:
                constant_(self.bv, 0.0)
            if self.b0 is not None:
                constant_(self.b0, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        # TODO: add (causallity) a mask for normal attention
        # but here it is a different function as implemented in the original paper
        # and for some reason q,k,v can have different dimensions this is not supported yet
        return rmha_forward(
            query,
            key,
            value,
            self.Wq,
            self.Wk,
            self.Wv,
            self.W0,
            self.num_heads,
            self.embed_dim,
            self.kernel_fn,
            self.causal,
            attention_mask=attention_mask,
            bq=self.bq,
            bk=self.bk,
            bv=self.bv,
            b0=self.b0,
            projection_matrix=self.projection_matrix,
            spre_model=self.srpe,
        ), None


def make_performer(
    mha: torch.nn.MultiheadAttention,
    num_random_features: int,
    kernel_fn: str,
    iscausal: bool,
) -> RandMultiHeadAttention:
    """
    Converts a standard PyTorch MultiheadAttention layer into a Performer-style
    random feature attention layer (RandMultiHeadAttention).

    Args:
        mha (torch.nn.MultiheadAttention): The input multi-head attention layer to convert.
        num_random_features (int): Number of random features to use in the Performer approximation.
        kernel_fn (str): Name of the kernel function to use for random feature mapping (e.g., 'relu', 'softmax').
        iscausal (bool): Whether the attention should be causal (prevents attending to future positions).

    Returns:
        RandMultiHeadAttention: A Performer-style attention layer with weights and biases copied from the input MHA.
    """
    embed_dim = mha.embed_dim
    num_heads = mha.num_heads
    dropout = mha.dropout
    bias = mha.in_proj_bias is not None
    performer = RandMultiHeadAttention(
        embed_dim,
        num_heads,
        num_random_features,
        dropout,
        bias,
        kernel_fn,
        iscausal,
    )
    performer.Wq = nn.Parameter(mha.in_proj_weight[:embed_dim, :])
    performer.Wk = nn.Parameter(mha.in_proj_weight[embed_dim : 2 * embed_dim, :])
    performer.Wv = nn.Parameter(mha.in_proj_weight[2 * embed_dim : 3 * embed_dim, :])
    performer.W0 = nn.Parameter(mha.out_proj.weight)

    if performer.bias:
        performer.bq = nn.Parameter(mha.in_proj_bias[:embed_dim])
        performer.bk = nn.Parameter(mha.in_proj_bias[embed_dim : 2 * embed_dim])
        performer.bv = nn.Parameter(mha.in_proj_bias[2 * embed_dim : 3 * embed_dim])
        performer.b0 = nn.Parameter(mha.out_proj.bias)

    return performer


if __name__ == "__main__":
    import os
    import time

    import psutil

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # helper to measure CPU memory if you ever need it
    def get_cpu_mem():
        p = psutil.Process(os.getpid())
        return p.memory_info().rss

    def benchmark_attention(
        attn_module, query, key, value, add_isCausal=False, iters=20, warmups=10
    ):
        attn_module = attn_module.to(device).eval()
        attn_mask = None
        if add_isCausal:
            # generate_square_subsequent_mask
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                query.shape[1], device=device
            ).to(device)
        # warm-up to JIT / kernel-cache everything
        with torch.no_grad():
            for _ in range(warmups):
                if add_isCausal:
                    _ = attn_module(
                        query, key, value, is_causal=True, attn_mask=attn_mask
                    )
                else:
                    _ = attn_module(query, key, value)

        times = []
        mems = []
        with torch.no_grad():
            for _ in range(iters):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize()
                else:
                    before = get_cpu_mem()

                start = time.time()
                if add_isCausal:
                    _ = attn_module(
                        query, key, value, is_causal=True, attn_mask=attn_mask
                    )
                else:
                    _ = attn_module(query, key, value)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - start

                if device.type == "cuda":
                    used = torch.cuda.max_memory_allocated(device)
                else:
                    used = get_cpu_mem() - before

                times.append(elapsed)
                mems.append(used)

        return sum(times) / len(times), sum(mems) / len(mems)

    # create some data
    batch, seq_len, dim = 1, 2048, 768
    num_heads = 12
    causal = False
    query = torch.randn(batch, seq_len, dim, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # instantiate exactly equivalent modules
    custom = RandMultiHeadAttention(
        embed_dim=dim,
        num_heads=num_heads,
        num_random_features=128,
        dropout=0.0,
        kernel_fn="softmax",
        iscausal=causal,
        device=device,
        dtype=torch.float32,
    )

    torch_mha = nn.MultiheadAttention(
        embed_dim=dim,
        num_heads=num_heads,
        dropout=0.0,
        batch_first=True,
        device=device,
        dtype=torch.float32,
    )

    # note: no masks, no need_weights, both non-causal
    # run benchmarks
    custom_t, custom_mem = benchmark_attention(
        custom, query, key, value, add_isCausal=False
    )
    torch_t, torch_mem = benchmark_attention(
        torch_mha, query, key, value, add_isCausal=causal
    )

    print(
        f"Custom RF-Attention —  avg time: {custom_t:.4f}s, avg mem: {custom_mem/1024:.1f} KB"
    )
    print(
        f" PyTorch MHA      —  avg time: {torch_t:.4f}s, avg mem: {torch_mem/1024:.1f} KB"
    )
