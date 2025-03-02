import torch

import lltm_cpp

d = lltm_cpp.Dog()
print(lltm_cpp.call_go(d))


class Cat(lltm_cpp.Animal):
    def go(self, n_times):
        torch.Tensor()
        return "meow! " * n_times


c = Cat()
print(lltm_cpp.call_go(c))
