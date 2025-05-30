#pragma once

#include <torch/extension.h>

#define AT_DISPATCH_CASE_FLOAT_AND_HALF(...)             \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define AT_DISPATCH_FLOAT_AND_HALF(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(                             \
        TYPE, NAME, AT_DISPATCH_CASE_FLOAT_AND_HALF(__VA_ARGS__))
