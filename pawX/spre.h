#pragma once

#include <torch/extension.h>

class Spre : public torch::
{
private:
    int num_heads;
    int perHead_in;
    int sines;
    int num_realizations;

public:
    Spre(int num_heads, int perHead_in, int sines, int num_realizations = 256)
        : num_heads(num_heads), perHead_in(perHead_in), sines(sines), num_realizations(num_realizations)
    {
    }
}