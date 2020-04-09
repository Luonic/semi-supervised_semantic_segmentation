import time

import torch
import torch.nn.functional as F
import reversible_augmentations
import losses
import metrics

import utils.utils as utils
import traceback

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

    # ce_criterion = losses.FocalLoss(alpha=0.75, gamma=0.25) # best for finetuning us a=0.75 g=0.25

    with torch.jit.optimized_execution(True):
        for step, sample in enumerate(dataloader):
            tic = time.time()
            global_step = initial_step + step
            image = sample['image'].to(device, non_blocking=True)
            # sup_unsup_image = torch.cat((image, unsupervised_image), dim=0)
            mask = sample['semantic_mask'].to(device, non_blocking=True)

            # if step == 0:
            #     model = torch.jit.trace(func=model,
            #                             example_inputs=image,
            #                             check_trace=True,
            #                             check_inputs=None,
            #                             check_tolerance=1e-5)

            pred_maps = model(image)
            # try:
            #     pred_maps = model(image)
            # except Exception as e:
            #     traceback.print_exc()
            #     print(sample['filename'])
            #     print(image.size())
            #     continue

            # sup_loss += losses.dice_loss(torch.sigmoid(pred_map_sup[:, 1:]), mask[:, 1:]).mean()
            # sup_loss = ce_criterion(pred_map_sup[:, 1:], mask[:, 1:]) * 2

            # loss = config['train']['loss'](pred_maps, losses.smooth_binary_labels(mask, alpha=0.2))
            loss = config['train']['loss'](pred_maps, mask)

            # Reduce loss from all workers
            reduced_loss = utils.reduce_tensor(loss) / world_size
            # if rank == 0:
            #     print('G loss:', loss.item())
            avg_loss.update(reduced_loss.item())

            optimizer.zero_grad()
            loss.backward()

            # for param in utils.get_trainable_params(model):
            #     if param.grad is None:
            #         print(param)

            # Unsupervised training
            unsupervised_image = next(unsupervised_dataloader)['image'].to(device, non_blocking=True)
            pred_maps = model(unsupervised_image)
            unsup_loss = sum(map(losses.entropy_loss, pred_maps)) * 0.001
            reduced_unsup_loss = utils.reduce_tensor(unsup_loss) / world_size
            avg_unsupervised_loss.update(reduced_unsup_loss.item())
            unsup_loss.backward()

            # if step + 1 % 4 == 0:
            # torch.nn.utils.clip_grad_norm_(model.module.parameters(), config['train']['gradient_clip_value'])
            optimizer.step()


            # Measure batch processing time
            batch_time.update(time.time() - tic)

            if step % config['train']['print_freq'] == 0 and rank == 0:
                msg = f'Epoch: {epoch} Step: {step} Batch time: {batch_time.average()} Loss: {avg_loss.average()}'
                print(msg)
                summary_writer.add_scalar('train_loss', reduced_loss.item(), global_step)
                summary_writer.add_scalar('train_unsupervised_loss', reduced_unsup_loss.item(), global_step)

        if rank == 0:
            summary_writer.add_scalar('batch_time', batch_time.average(), global_step)
            summary_writer.add_scalar('train_loss_avg', avg_loss.average(), global_step)
            summary_writer.add_scalar('train_unsupervised_loss_avg', avg_unsupervised_loss.average(), global_step)


def validate(model, dataloader, epoch, initial_step, summary_writer, config, device):
    model.eval()
    with torch.no_grad():
        avg_loss = utils.AverageMeter()
        avg_metric = utils.AverageMeter()
        batch_time = utils.AverageMeter()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        tic = time.time()

        if rank == 0:
            print('Evaluating...')

        for sample in dataloader:
            image = sample['image'].to(device)
            mask = sample['semantic_mask'].to(device)

            # pred_map = model(image)['out']
            pred_maps = model(image)

            # loss = config['train']['loss'](pred_maps, losses.smooth_binary_labels(mask, alpha=0.2))
            loss = config['train']['loss'](pred_maps, mask)
            one_hot_prediction = torch.nn.functional.one_hot(torch.argmax(pred_maps[-1], dim=1), num_classes=2).permute(dims=(0, 3, 1, 2)).to(pred_maps[0])
            pred_map_binary = torch.nn.functional.interpolate(one_hot_prediction, size=mask.size()[2:4], mode='nearest')
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
