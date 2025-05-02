import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from panther.nn.pawXimpl import create_projection_matrix, rmha_forward


def verify_rmha_inputs(
    embed_dim,
    num_heads,
    dropout,
    bias,
    kernel_fn,
    iscausal,
):  # verfies all the performers inputs
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

        self.register_buffer(
            "projection_matrix",
            create_projection_matrix(
                num_random_features, embed_dim // num_heads, **factory_kwargs
            ),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
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
        ), None


def make_performer(
    mha: torch.nn.MultiheadAttention,
    num_random_features: int,
    kernel_fn: str,
    iscausal: bool,
) -> RandMultiHeadAttention:
    """
    Converts a MultiheadAttention layer into a Performers layer.
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RandMultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        num_random_features=10,
        dropout=0.1,
        kernel_fn="softmax",
        iscausal=False,
        device=device,
        dtype=torch.float32,
    ).to(device)

    output = model(
        query=torch.randn(1, 2, 8).to(device),
        key=torch.randn(1, 3, 8).to(device),
        value=torch.randn(1, 3, 8).to(device),
        attention_mask=torch.tensor([[[0, 0, 0], [0, 0, 0]]], dtype=torch.bool).to(
            device
        ),
        # attention_mask=None,
    )
    print(output[0].shape)
    print(output[0])
