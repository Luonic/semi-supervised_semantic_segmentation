import time

import torch
import torch.nn.functional as F
import reversible_augmentations
import s4gan
import losses

import utils.utils as utils


# Write differentiable augmentations (rotations and scaling) using kornia
# Sample randomly 2 aug params, do forward, warp preds back, calc consistency loss and do backward
# Train in switchable mode: one step of supervised and multiple steps of semi-supervised

# 1) Write optimized hrnet model
# 2) Steal from hrnet LIP dataset loader and modify it
# 3) Pretrain custom hrnet with lip dataset
# 4) Finetune HRNet with my dataset

def train(model, discriminator, optimizer, discriminator_optimizer, dataloader, unsupervised_dataloader, epoch,
          initial_step, summary_writer, config, device):
    model.train()
    discriminator.train()
    avg_loss = utils.AverageMeter()
    avg_unsupervised_loss = utils.AverageMeter()
    avg_self_training_mask_mean = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    avg_discriminator_loss = utils.AverageMeter()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    ce_criterion = losses.FocalLoss(alpha=1, gamma=5.0)

    for step, sample in enumerate(dataloader):
        tic = time.time()
        global_step = initial_step + step
        image = sample['image'].to(device, non_blocking=True)
        unsupervised_image = next(unsupervised_dataloader)['image'].to(device)
        sup_unsup_image = torch.cat((image, unsupervised_image), dim=0)
        mask = sample['semantic_mask'].to(device, non_blocking=True)

        # pred_map = model(sup_unsup_image)['out']
        pred_map = model(sup_unsup_image)
        pred_map = torch.nn.functional.interpolate(pred_map, size=(image.size(2), image.size(3)), mode='bilinear')

        pred_map_sup, pred_map_unsup = torch.split(pred_map, split_size_or_sections=image.size(0), dim=0)

        # Discriminator training
        if global_step > config['train']['semi_supervised_training_start_step']:
            pred_d_real = discriminator(torch.cat((image, mask), dim=1).detach())
            pred_d_fake = discriminator(torch.cat((unsupervised_image, torch.sigmoid(pred_map_unsup)), dim=1).detach())

            # discriminator_loss = -(torch.log(pred_d_real).mean() + torch.log(1 - pred_d_fake).mean()) / 2

            discriminator_loss = (torch.nn.functional.binary_cross_entropy_with_logits(pred_d_real,
                                                                                       torch.ones_like(pred_d_real)) +
                                  torch.nn.functional.binary_cross_entropy_with_logits(pred_d_fake,
                                                                                       torch.zeros_like(
                                                                                           pred_d_fake))) / 2.

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.module.parameters(), config['train']['gradient_clip_value'])
            discriminator_optimizer.step()

            reduced_discrimiantor_loss = utils.reduce_tensor(discriminator_loss) / world_size
            avg_discriminator_loss.update(reduced_discrimiantor_loss.item())
            # if rank == 0:
            #     print('D loss:', discriminator_loss.item())
        else:
            avg_discriminator_loss.update(0)

        # Segmentation net training
        # sup_loss = (ce_criterion(pred_map_sup, mask) +
        #             torch.nn.functional.binary_cross_entropy_with_logits(pred_map_sup, mask) +
        #             losses.dice_loss(torch.sigmoid(pred_map_sup[:, 1:]), mask[:, 1:]).mean() * 0.5)
        # print('supervised_probabilities', pred_map_sup.min().item(), pred_map_sup.max().item())
        sup_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_map_sup, mask).mean()

        if global_step > config['train']['semi_supervised_training_start_step']:
            pred_d_real = discriminator(torch.cat((image, mask), dim=1))
            pred_d_fake_sup = discriminator(torch.cat((image, torch.sigmoid(pred_map_sup)), dim=1))
            pred_d_fake_unsup = discriminator(torch.cat((unsupervised_image, torch.sigmoid(pred_map_unsup)), dim=1))

            supervised_adversarial_loss = \
                torch.nn.functional.binary_cross_entropy_with_logits(pred_d_fake_sup,
                                                                     torch.ones_like(pred_d_fake_sup))
            unsupervised_adversarial_loss = \
                torch.nn.functional.binary_cross_entropy_with_logits(pred_d_fake_unsup,
                                                                     torch.ones_like(pred_d_fake_unsup))

            indicator = (torch.sigmoid(pred_d_fake_unsup) > config['train']['semi_supervised_threshold']).to(
                pred_d_fake_unsup)
            indicator = torch.nn.functional.interpolate(indicator, (pred_map_unsup.size(2), pred_map_unsup.size(3)),
                                                        mode='bilinear', align_corners=True)
            pseudolabel = (torch.sigmoid(pred_map_unsup) > 0.5).to(pred_map_unsup)
            # print('unsupervised_probabilities', pred_map_unsup.min().item(), pred_map_unsup.max().item())
            semi_supervised_loss = (indicator *
                                    torch.nn.functional.binary_cross_entropy_with_logits(pred_map_unsup,
                                                                                         pseudolabel,
                                                                                         reduction='none')).sum() / \
                                   (indicator.sum() + 0.0001)

            reduced_self_train_mask_mean = utils.reduce_tensor(indicator.mean()) / world_size
            avg_self_training_mask_mean.update(reduced_self_train_mask_mean.item())
        else:
            supervised_adversarial_loss = 0
            unsupervised_adversarial_loss = 0
            semi_supervised_loss = 0

            avg_self_training_mask_mean.update(0)

        loss = (sup_loss +
                supervised_adversarial_loss * config['train']['supervised_adversarial_loss_weight'] +
                unsupervised_adversarial_loss * config['train']['unsupervised_adversarial_loss_weight'] +
                semi_supervised_loss * config['train']['semi_supervised_loss_weight']).mean()

        # loss = sup_loss

        # Reduce loss from all workers
        reduced_loss = utils.reduce_tensor(loss) / world_size
        # if rank == 0:
        #     print('G loss:', loss.item())
        avg_loss.update(reduced_loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.module.parameters(), config['train']['gradient_clip_value'])
        optimizer.step()

        # Measure batch processing time
        batch_time.update(time.time() - tic)

        if step % config['train']['print_freq'] == 0 and rank == 0:
            msg = f'Epoch: {epoch} Step: {step} Batch time: {batch_time.average()} Loss: {avg_loss.average()}'
            print(msg)
            summary_writer.add_scalar('train_loss', reduced_loss, global_step)

    if rank == 0:
        summary_writer.add_scalar('batch_time', batch_time.average(), global_step)
        summary_writer.add_scalar('train_loss_avg', avg_loss.average(), global_step)
        # summary_writer.add_scalar('train_unsupervised_loss_avg', avg_unsupervised_loss.average(), global_step)
        summary_writer.add_scalar('train_discriminator_loss_avg', avg_discriminator_loss.average(), global_step)
        summary_writer.add_scalar('train_self_training_mask_mean', avg_self_training_mask_mean.average(), global_step)


def validate(model, dataloader, epoch, initial_step, summary_writer, config, device):
    model.eval()
    with torch.no_grad():
        avg_loss = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        tic = time.time()

        for sample in dataloader:
            image = sample['image'].to(device)
            mask = sample['semantic_mask'].to(device)

            # pred_map = model(image)['out']
            pred_map = model(image)
            pred_map = torch.nn.functional.interpolate(pred_map, size=(image.size(2), image.size(3)), mode='bilinear')

            # loss = torch.abs(pred_map - mask).mean()
            # mask = (mask > 0.5).to(mask)
            loss = F.binary_cross_entropy_with_logits(pred_map, mask)

            # Reduce loss from all workers
            reduced_loss = utils.reduce_tensor(loss) / world_size
            avg_loss.update(reduced_loss.item())

            # measure batch processing time
            batch_time.update(time.time() - tic)
            tic = time.time()

    if rank == 0:
        msg = f'Eval: Epoch: {epoch} Batch time: {batch_time.average()} Val loss: {avg_loss.average()}'
        print(msg)
        summary_writer.add_scalar('val_loss_avg', avg_loss.average(), initial_step)

    return avg_loss.average()

# if train_encoder:
#     trainable_params = model.parameters(recurse=True)
# else:
#     trainable_params = list(
#         set(list(model.parameters(recurse=True))) - set(list(model.encoder.parameters(recurse=True))))
