import torch
import torch.nn as nn


class SKLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_terms: int,
        low_rank: int,
        bias_init_std=0.01,
        W=None,
    ):
        super(SKLinear, self).__init__()

        if 2 * num_terms * low_rank * (output_dim + input_dim) > output_dim * input_dim:
            raise ValueError(
                "The number of parameters in the sketching layer is larger "
                + "than the number of parameters in the fully connected layer."
            )

        self.l = num_terms
        self.k = low_rank
        self.d1 = output_dim
        self.d2 = input_dim

        def generate_U(k: int, d: int) -> torch.Tensor:
            """
            Generate a random matrix U with orthonormal rows.
            """
            return (
                torch.randint(0, 2, (k, d), dtype=torch.float32) * 2 - 1
            ) / torch.sqrt(torch.tensor(k, dtype=torch.float32))

        # Register U1s and U2s as buffers since they are not learnable
        self.register_buffer(
            "U1s",
            torch.tensor(
                [generate_U(low_rank, output_dim) for _ in range(num_terms)],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "U2s",
            torch.tensor(
                [generate_U(low_rank, input_dim) for _ in range(num_terms)],
                dtype=torch.float32,
            ),
        )

        # W is learnable, initialized randomly if not provided
        self.W = nn.Parameter(
            torch.randn(output_dim, input_dim)
            if W is None
            else torch.tensor(W, dtype=torch.float32)
        )

        # S1s and S2s are precomputed but not updated in the backward pass
        self.S1s = [torch.matmul(self.U1s[i], self.W) for i in range(num_terms)]
        self.S2s = [torch.matmul(self.W, self.U2s[i].T) for i in range(num_terms)]

        # Bias term initialized with a small standard deviation
        self.b = nn.Parameter(torch.randn(output_dim) * bias_init_std)

    def forward(self, h_in):
        """
        Forward pass calculation as per the given formula:
        a = (1/(2*l)) * sum_{i=1}^{l} (U1_i^T S1_i h_in)
            + (1/(2*l)) * sum_{i=1}^{l} (S2_i U2_i h_in)
            + b
        """
        tot = torch.zeros(self.d1, device=h_in.device)

        # Efficiently perform the sum over all l terms
        for i in range(self.l):
            tot += torch.matmul(
                self.U1s[i].T, torch.matmul(self.S1s[i], h_in)
            ) + torch.matmul(self.S2s[i], torch.matmul(self.U2s[i], h_in))

        return tot / (2 * self.l) + self.b
