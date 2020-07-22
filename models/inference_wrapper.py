import torch
import torch.nn as nn

class InferenceWrapper(nn.Module):
    def __init__(self, model):
        super(InferenceWrapper, self).__init__()
        self.model = model

    def forward(self, image):
        # image: 3d image of shape CxHxW
        image = torch.unsqueeze(image, 0)
        features, logit_resolutions = self.model(image)
        high_resolution_logits = logit_resolutions[0]
        high_resolution_logits = torch.nn.functional.interpolate(high_resolution_logits,
                                                                 size=[image.size(2), image.size(3)],
                                                                 mode='bilinear',
                                                                 align_corners=False)
        # probabilities = torch.softmax(high_resolution_logits, dim=1)
        probabilities = torch.sigmoid(high_resolution_logits)
        class_indices = torch.argmax(high_resolution_logits, dim=1)
        binary_mask_nhwc = torch.nn.functional.one_hot(class_indices, num_classes=2)
        binary_mask_nchw = binary_mask_nhwc.permute(dims=(0, 3, 1, 2)).to(probabilities)
        probabilities_3d = probabilities[0]
        binary_mask_3d = binary_mask_nchw[0]
        return binary_mask_3d, probabilities_3d
