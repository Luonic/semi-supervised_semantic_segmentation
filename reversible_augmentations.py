import torch
import kornia


class Rotate:
    def __init__(self, max_angle):
        self.max_angle = torch.abs(torch.tensor([max_angle], dtype=torch.float32))
        self.distribution = torch.distributions.uniform.Uniform(low=-self.max_angle, high=self.max_angle)
        self.angle = None

    def apply(self, inputs):
        result_list = []

        self.angle = self.distribution.sample()
        for input_tensor in inputs:
            result_list.append(kornia.rotate(input_tensor, self.angle.to(input_tensor.device)))
        return result_list

    def reverse(self, inputs):
        result_list = []
        for input_tensor in inputs:
            result_list.append(kornia.rotate(input_tensor, - self.angle.to(input_tensor.device)))
        return result_list


class Rescale:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.distribution = torch.distributions.uniform.Uniform(low=torch.tensor([min_scale], dtype=torch.float32),
                                                                high=torch.tensor([max_scale], dtype=torch.float32))
        self.scale = None

    def apply(self, inputs):
        result_list = []
        self.scale = self.distribution.sample().item()
        for input_tensor in inputs:
            size = (int(input_tensor.size(2) * self.scale),
                    int(input_tensor.size(3) * self.scale))
            result_list.append(kornia.resize(input_tensor, size))
        return result_list

    def reverse(self, inputs):
        result_list = []
        for input_tensor in inputs:
            size = (int(input_tensor.size(2) * (1. / self.scale)),
                    int(input_tensor.size(3) * (1. / self.scale)))
            result_list.append(kornia.resize(input_tensor, size))
        return result_list
