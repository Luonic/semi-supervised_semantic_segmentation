import time

import torch
import torch.nn.functional as F
import reversible_augmentations
import losses
import metrics

import utils.utils as utils


# Write differentiable augmentations (rotations and scaling) using kornia
# Sample randomly 2 aug params, do forward, warp preds back, calc consistency loss and do backward
# Train in switchable mode: one step of supervised and multiple steps of semi-supervised

# 1) Write optimized hrnet model
# 2) Steal from hrnet LIP dataset loader and modify it
# 3) Pretrain custom hrnet with lip dataset
# 4) Finetune HRNet with my dataset

def train(model, optimizer, dataloader, unsupervised_dataloader, epoch,
          initial_step, summary_writer, config, device):
    model.train()
    avg_loss = utils.AverageMeter()
    avg_unsupervised_loss = utils.AverageMeter()
    avg_self_training_mask_mean = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    avg_discriminator_loss = utils.AverageMeter()

    lr = utils.get_max_lr(optimizer)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    ce_criterion = losses.FocalLoss(alpha=0.75, gamma=0.25) # best for finetuning us a=0.75 g=0.25

    for step, sample in enumerate(dataloader):
        tic = time.time()
        global_step = initial_step + step
        image = sample['image'].to(device, non_blocking=True)
        # unsupervised_image = next(unsupervised_dataloader)['image'].to(device)
        # sup_unsup_image = torch.cat((image, unsupervised_image), dim=0)
        mask = sample['semantic_mask'].to(device, non_blocking=True)

        pred_maps = model(image)

        # sup_loss += losses.dice_loss(torch.sigmoid(pred_map_sup[:, 1:]), mask[:, 1:]).mean()
        # sup_loss = ce_criterion(pred_map_sup[:, 1:], mask[:, 1:]) * 2

        loss = config['train']['loss'](map(lambda x: x[:, 1:], pred_maps), mask[:, 1:])

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


def validate(model, dataloader, epoch, initial_step, summary_writer, config, device):
    model.eval()
    with torch.no_grad():
        avg_loss = utils.AverageMeter()
        avg_metric = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        tic = time.time()

        for sample in dataloader:
            image = sample['image'].to(device)
            mask = sample['semantic_mask'].to(device)

            # pred_map = model(image)['out']
            pred_maps = model(image)

            loss = config['train']['loss'](map(lambda x: x[:, 1:], pred_maps), mask[:, 1:])

            pred_map_binary = (torch.sigmoid(pred_maps[-1]) > 0.5).to(pred_maps[0])
            mask_binary = (mask > 0.5).to(mask)
            metric = metrics.dice_metric(pred_map_binary[:, 1:], mask_binary[:, 1:]).mean()

            # Reduce loss from all workers
            reduced_loss = utils.reduce_tensor(loss) / world_size
            avg_loss.update(reduced_loss.item())

            reduced_metric = utils.reduce_tensor(metric) / world_size
            avg_metric.update(reduced_metric.item())

            # measure batch processing time
            batch_time.update(time.time() - tic)
            tic = time.time()

    if rank == 0:
        msg = f'Eval: Epoch: {epoch} Batch time: {batch_time.average()} Val loss: {avg_loss.average()} ' \
              f'Val metric: {avg_metric.average()}'
        print(msg)
        summary_writer.add_scalar('val_loss_avg', avg_loss.average(), initial_step)
        summary_writer.add_scalar('val_dice', avg_metric.average(), initial_step)

    return avg_loss.average(), avg_metric.average()
