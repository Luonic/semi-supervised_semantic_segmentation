import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torchvision
torchvision.models.segmentation.deeplabv3_resnet101()

def deeplabv3_resnet101(num_classes):
    model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
    model.classifier[-1] = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
    del model.aux_classifier

    import types

    def get_params_with_layerwise_lr(model, base_lr):
        params = []
        # params.append({'params': model.backbone.parameters(), 'lr': base_lr * 0.1})
        params.append({'params': model.backbone.parameters(), 'lr': 0})
        last_layer_params = model.classifier[-1].parameters()
        decoder_except_last_layer_params = list(set(model.classifier.parameters()) -
                                                set(last_layer_params))
        params.append({'params': decoder_except_last_layer_params, 'lr': base_lr})
        params.append({'params': last_layer_params, 'lr': base_lr})
        return params

    model.get_params_with_layerwise_lr = get_params_with_layerwise_lr

    def custom_forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

    model.forward = types.MethodType(custom_forward, model)

    return model

def deeplabv3_resnet50(num_classes):
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
    model.classifier[-1] = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
    del model.aux_classifier

    import types

    def get_params_with_layerwise_lr(model, base_lr):
        params = []
        # params.append({'params': model.backbone.parameters(), 'lr': base_lr * 0.1})
        params.append({'params': model.backbone.parameters(), 'lr': 0})
        last_layer_params = model.classifier[-1].parameters()
        decoder_except_last_layer_params = list(set(model.classifier.parameters()) -
                                                set(last_layer_params))
        params.append({'params': decoder_except_last_layer_params, 'lr': base_lr})
        params.append({'params': last_layer_params, 'lr': base_lr})
        return params

    model.get_params_with_layerwise_lr = get_params_with_layerwise_lr

    def custom_forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

    model.forward = types.MethodType(custom_forward, model)

    return model

def fcn_resnet50(num_classes):
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False)
    model.classifier[-1] = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
    del model.aux_classifier

    import types

    def get_params_with_layerwise_lr(model, base_lr):
        params = []
        # params.append({'params': model.backbone.parameters(), 'lr': base_lr * 0.1})
        params.append({'params': model.backbone.parameters(), 'lr': 0})
        last_layer_params = model.classifier[-1].parameters()
        decoder_except_last_layer_params = list(set(model.classifier.parameters()) -
                                                set(last_layer_params))
        params.append({'params': decoder_except_last_layer_params, 'lr': base_lr})
        params.append({'params': last_layer_params, 'lr': base_lr})
        return params

    model.get_params_with_layerwise_lr = get_params_with_layerwise_lr

    def custom_forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

    model.forward = types.MethodType(custom_forward, model)

    return model
