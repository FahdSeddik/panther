import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_


def causal_numerator(
    query_prime: torch.Tensor, key_prime: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    return torch.tensor([0])


def causal_denominator(
    query_prime: torch.Tensor, key_prime: torch.Tensor
) -> torch.Tensor:
    return torch.tensor([0])


def noncausal_numerator(
    query_prime: torch.Tensor, key_prime: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    kvs = torch.matmul(key_prime.permute(1, 2, 3, 0), value.permute(1, 2, 0, 3))
    return torch.matmul(query_prime.permute(1, 2, 0, 3), kvs).permute(2, 0, 1, 3)


def noncausal_denominator(
    query_prime: torch.Tensor, key_prime: torch.Tensor
) -> torch.Tensor:
    ks_sum = key_prime.sum(dim=0)
    return (query_prime * ks_sum).sum(dim=-1)


def create_projection_matrix(m: int, d, seed=42, scaling=False) -> torch.Tensor:
    torch.manual_seed(seed)  # Set seed for reproducibility

    nb_full_blocks = m // d  # Number of complete dxd blocks
    block_list = []  # List to store blocks
    # does QR on steps which is more effcient and numerically stable and guarantees orthogonality
    for _ in range(nb_full_blocks):
        # Generate a random dxd Gaussian matrix
        unstructured_block = torch.randn(d, d)
        # Perform QR decomposition to obtain an orthonormal matrix
        q, _ = torch.linalg.qr(unstructured_block)
        q = q.T  # Transpose so rows are the orthonormal vectors

        block_list.append(q)  # Store the block

    remaining_rows = m - nb_full_blocks * d  # Compute remaining rows

    if remaining_rows > 0:
        # Handle the last incomplete block
        unstructured_block = torch.randn(d, d)
        q, _ = torch.linalg.qr(unstructured_block)
        q = q.T  # Transpose
        block_list.append(q[:remaining_rows])  # Take only required rows

    # Stack all blocks to form the final projection matrix
    final_matrix = torch.vstack(block_list)

    if scaling:
        # If scaling is enabled, normalize rows to sqrt(d)
        multiplier = torch.full((m,), torch.sqrt(torch.tensor(d)))
    else:
        # Otherwise, scale each row using chi(d) distribution
        multiplier = torch.norm(torch.randn(m, d), dim=1)

    # Apply scaling by multiplying with diagonal matrix
    return torch.matmul(torch.diag(multiplier), final_matrix)


def relu_kernel_transform(
    data: torch.Tensor,
    is_query=True,
    projection_matrix: torch.Tensor | None = None,
    numerical_stabilizer=0.00001,
):
    if projection_matrix is None:
        return torch.nn.functional.relu(data) + numerical_stabilizer
    else:
        ratio = 1.0 / torch.sqrt(
            torch.tensor(projection_matrix.shape[0], dtype=torch.float32)
        )
        data_dash = ratio * torch.matmul(
            data, projection_matrix.T
        )  # Equivalent to einsum("blhd,md->blhm")
        return torch.nn.functional.relu(data_dash) + numerical_stabilizer


def softmax_kernel_transform(
    data: torch.Tensor,
    is_query=True,
    projection_matrix: torch.Tensor | None = None,
    numerical_stabilizer=0.00001,
) -> torch.Tensor:
    if projection_matrix is None:
        raise ValueError("projection_matrix is required for softmax kernel")
    data_normalizer = 1.0 / (torch.math.sqrt(torch.math.sqrt(data.shape[-1])))
    data = data_normalizer * data
    ratio = 1.0 / torch.math.sqrt(projection_matrix.shape[0])  # 1/sqrt(m)
    data_dash = torch.matmul(data, projection_matrix.T)
    diag_data = (
        data**2
    )  # basically ||x||^2/2 for all x in seq_length in all batches (the h(x) of the kernel)
    diag_data = diag_data.sum(dim=-1)
    diag_data = diag_data / 2.0
    diag_data = diag_data.unsqueeze(dim=data.ndim - 1)  # prodcasting purposes
    max_val = data_dash.max(dim=-1, keepdim=True).values  # (batch, seq_len, heads, 1)
    if not is_query:
        max_val = max_val.max(dim=-3, keepdim=True).values  # (batch, 1, heads, 1)
    # here is should multiply by diag_data, i dont know why it subs though
    return ratio * (torch.exp(data_dash - diag_data - max_val) + numerical_stabilizer)


def favor_attention(query, key, value, kernel_fn: str, causal, projection_matrix=None):
    if kernel_fn == "relu":
        query_prime: torch.Tensor = relu_kernel_transform(
            query, True, projection_matrix
        )
        key_prime: torch.Tensor = relu_kernel_transform(key, False, projection_matrix)
    elif kernel_fn == "softmax":
        query_prime = softmax_kernel_transform(query, True, projection_matrix)
        key_prime = softmax_kernel_transform(key, False, projection_matrix)
    query_prime = query_prime.permute(1, 0, 2, 3)
    key_prime = key_prime.permute(1, 0, 2, 3)
    value = value.permute(1, 0, 2, 3)
    if causal:
        av_attention = causal_numerator(query_prime, key_prime, value)
        attention_normalizer = causal_denominator(query_prime, key_prime)
    else:
        av_attention = noncausal_numerator(query_prime, key_prime, value)
        attention_normalizer = noncausal_denominator(query_prime, key_prime)
    av_attention = av_attention.permute(1, 0, 2, 3)
    attention_normalizer = attention_normalizer.permute(1, 0, 2).unsqueeze(-1)
    return av_attention / attention_normalizer


# TODO: backpropagation is not implemented yet
class Performers(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_random_features,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ):
        super(Performers, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.dropout = dropout
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.kdim = kdim
        self.vdim = vdim
        self.batch_first = batch_first
        self.device = device
        self.dtype = dtype
        self.Wq = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wk = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wv = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W0 = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.bq = None
        self.bk = None
        self.bv = None
        self.b0 = None
        if bias:
            self.bq = nn.Parameter(torch.randn(embed_dim))
            self.bk = nn.Parameter(torch.randn(embed_dim))
            self.bv = nn.Parameter(torch.randn(embed_dim))
            self.b0 = nn.Parameter(torch.randn(embed_dim))
        self.training = True
        self.kernel_fn = "softmax"
        self.causal = False
        self.register_buffer(
            "projection_matrix",
            create_projection_matrix(num_random_features, embed_dim // num_heads),
        )
        xavier_uniform_(self.Wq)
        xavier_uniform_(self.Wk)
        xavier_uniform_(self.Wv)
        if bias:
            constant_(self.b0, 0.0)
            constant_(self.bk, 0.0)
            constant_(self.bq, 0.0)
            constant_(self.bv, 0.0)

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: add (causallity) a mask for normal attention
        # but here it is a different function as implemented in the original paper
        # and for some reason q,k,v can have different dimensions this is not supported yet
        query = nn.functional.linear(query, self.Wq, self.bq)
        key = nn.functional.linear(key, self.Wk, self.bk)
        value = nn.functional.linear(value, self.Wv, self.bv)
        query = query.view(
            query.shape[0],
            query.shape[1],
            self.num_heads,
            self.embed_dim // self.num_heads,
        )
        key = key.view(
            key.shape[0], key.shape[1], self.num_heads, self.embed_dim // self.num_heads
        )
        value = value.view(
            value.shape[0],
            value.shape[1],
            self.num_heads,
            self.embed_dim // self.num_heads,
        )
        attention_output = favor_attention(
            query, key, value, self.kernel_fn, self.causal, self.projection_matrix
        )
        # this following part can be optimized,
        # reshape->view, have attention shape as (batch*seq_len,embed_dim) before using linear
        attention_output = attention_output.reshape(
            query.shape[0], query.shape[1], self.embed_dim
        )
        attention_output = nn.functional.linear(attention_output, self.W0, self.b0)
        return attention_output
