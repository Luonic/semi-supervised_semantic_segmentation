import torch
import torch.nn as nn
import torch.nn.functional as F

# Training on labelled data
def joint_probability(probability, target):
    # probability: latent branch probabilities of shape NxLxHxW. L is a number of latent classes
    # target: ground-truth one-hot encoded semantic maps of shape NxCxHxW. C is a number of semantic classes
    #
    flattened_probability = torch.flatten(probability.permute(0, 2, 3, 1), start_dim=0, end_dim=-2)
    flattened_target = torch.flatten(target.permute(0, 2, 3, 1), start_dim=0, end_dim=-2)
    return torch.matmul(flattened_probability.t(), flattened_target) / \
           (flattened_probability.size(0)).type(probability.type())


def conditional_probability(joint_probability):
    # joint_probability: LxC matrix. L is number of latent classes and C is number of  semantic classes
    denominator = joint_probability.sum(dim=2, keepdim=True)
    return joint_probability / denominator


def latent_loss(probability, target):
    # probability: latent branch probabilities of shape NxLxHxW. L is a number of latent classes
    # target: ground-truth one-hot encoded semantic maps of shape NxCxHxW. C is a number of semantic classes
    joint_prob = joint_probability(probability, target)
    conditional_prob = conditional_probability(joint_prob)
    return - (joint_prob * torch.log(conditional_prob)).sum()


def semantic_cross_entropy_loss(probability, target):
    # probability: semantic branch probabilities of shape NxCxHxW. C is a number of semantic classes
    # target: ground-truth one-hot encoded semantic maps of shape NxCxHxW. C is a number of semantic classes

    return F.nll_loss(input=probability, target=target, reduction='mean')


def semantic_adversarial_loss(discriminator_pred_fake):
    # discriminator_pred_fake: probability map predicted from discriminator of size Nx1xHxW
    return F.nll_loss(discriminator_pred_fake,  torch.ones_like(discriminator_pred_fake), reduction='mean')


def total_labelled_loss(latent_probability, semantic_probability, discriminator_pred_fake, semantic_taget,
                        latent_loss_weight, adversarial_loss_weight):
    return (semantic_cross_entropy_loss(semantic_probability, semantic_taget) +
            latent_loss_weight * latent_loss(latent_probability, semantic_taget) +
            adversarial_loss_weight * semantic_adversarial_loss(discriminator_pred_fake))

#  Trainig on unlabelled data

