import torch
import torch.nn as nn


def apply_spre(
    query: torch.Tensor,
    key: torch.Tensor,
    qbar: torch.Tensor,
    kbar: torch.Tensor,
):
    print(qbar.shape, query.shape)
    qhat = (qbar * query.unsqueeze(-1)).sum(dim=-2)
    khat = (kbar * key.unsqueeze(-1)).sum(dim=-2)
    return qhat, khat


class sinSRPE(nn.Module):
    freqs: torch.nn.Parameter
    phases: torch.nn.Parameter
    scales: torch.nn.Parameter
    z: torch.nn.Parameter

    def __init__(
        self,
        num_heads: int,
        perHead_in: int,
        sines: int,
        num_realizations=256,
        device=None,
        dtype=None,
    ):
        super(sinSRPE, self).__init__()
        self.num_heads = num_heads
        self.perHead_in = perHead_in
        self.num_realizations = num_realizations
        self.sines = sines
        self.device = device
        self.dtype = dtype

        self.register_parameter(
            "freqs",
            torch.nn.Parameter(
                torch.randn(num_heads, perHead_in, sines, device=device, dtype=dtype)
            ),
        )
        self.register_parameter(
            "phases",
            torch.nn.Parameter(
                torch.randn(num_heads, perHead_in, sines, device=device, dtype=dtype)
            ),
        )
        self.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.randn(num_heads, perHead_in, sines, device=device, dtype=dtype)
            ),
        )
        self.register_parameter(
            "z",
            torch.nn.Parameter(
                torch.randn(
                    1,
                    num_heads,
                    perHead_in,
                    2 * sines,
                    num_realizations,
                    device=device,
                    dtype=dtype,
                )
            ),
        )

        self.scales.data /= torch.sqrt(self.scales.norm(dim=-1, keepdim=True)) / 2.0
        self.freqs.data -= 4.0

    def forward(self, len: int):
        indices = (
            torch.arange(0, len, device=self.device, dtype=self.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        freqs = torch.sigmoid(self.freqs) / 2
        freqs = freqs.unsqueeze(-2)

        phases_q = 2 * torch.pi * indices * freqs + self.phases.unsqueeze(-2)
        omega_q = torch.cat(
            [torch.cos(phases_q), torch.sin(phases_q)], dim=-1
        ).unsqueeze(0)
        phases_k = 2 * torch.pi * indices * freqs
        omega_k = torch.cat(
            [torch.cos(phases_k), torch.sin(phases_k)], dim=-1
        ).unsqueeze(0)
        scales = nn.functional.softplus(self.scales)

        scales = torch.cat([scales, scales], dim=-1)
        scales = scales.unsqueeze(0).unsqueeze(-1)

        z = self.z * scales
        qbar = torch.matmul(omega_q, z)
        kbar = torch.matmul(omega_k, z)
        qbar = qbar.permute(0, 3, 1, 2, 4)
        kbar = kbar.permute(0, 3, 1, 2, 4)
        scale = (self.num_realizations * self.perHead_in) ** 0.25
        qbar = qbar / scale
        kbar = kbar / scale
        return qbar, kbar


if __name__ == "__main__":
    length = 10
    num_heads = 8
    perHead_in = 128
    batch_size = 2
    srpe = sinSRPE(num_heads, perHead_in, 16)
    qbar, kbar = srpe(length)

    query = torch.randn(batch_size, length, num_heads, perHead_in)
    key = torch.randn(batch_size, length, num_heads, perHead_in)
    qhat, khat = apply_spre(query, key, qbar, kbar)
    print(qhat.shape, khat.shape)
