import time

import torch
import torch.nn.functional as F
import reversible_augmentations
import losses
import metrics
import cowmix
from tqdm import tqdm

import utils.utils as utils
import traceback
import mean_teacher


# Write differentiable augmentations (rotations and scaling) using kornia
# Sample randomly 2 aug params, do forward, warp preds back, calc consistency loss and do backward
# Train in switchable mode: one step of supervised and multiple steps of semi-supervised

# 1) Write optimized hrnet model
# 2) Steal from hrnet LIP dataset loader and modify it
# 3) Pretrain custom hrnet with lip dataset
# 4) Finetune HRNet with my dataset

def train(model, ema_model, optimizer, dataloader, unsupervised_dataloader, epoch,
          initial_step, summary_writer, config, device):
    model.train()
    avg_classification_loss = utils.AverageMeter()
    avg_supervised_loss = utils.AverageMeter()
    avg_unsupervised_loss = utils.AverageMeter()
    avg_confidence_modulator = utils.AverageMeter()

    batch_time = utils.AverageMeter()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    optimizer.zero_grad()

    with torch.jit.optimized_execution(True):
        for step, sample in enumerate(dataloader):
            tic = time.time()
            global_step = initial_step + step
            image = sample['image'].to(device, non_blocking=True)
            mask = sample['semantic_mask'].to(device, non_blocking=True)

            features, pred_maps = model(image)

            # loss = config['train']['loss'](pred_maps, losses.smooth_binary_labels(mask, alpha=0.2))
            # classification_loss = config['train']['loss'](pred_maps, losses.smooth_labels(mask, alpha=0.05))
            classification_loss = config['train']['loss'](pred_maps, mask)
            # Reduce classification loss from all workers
            reduced_classification_loss = utils.reduce_tensor(classification_loss) / world_size
            avg_classification_loss.update(reduced_classification_loss.item())

            # Loss for supervised step of training
            sup_loss = classification_loss
            reduced_sup_loss = utils.reduce_tensor(sup_loss) / world_size
            avg_supervised_loss.update(reduced_sup_loss.item())

            (sup_loss / config['train']['virtual_batch_size_multiplier']).backward()
            del pred_maps

            # Unsupervised training
            if False:
                unsupervised_image_a = next(unsupervised_dataloader)['image'].to(device, non_blocking=True)
                unsupervised_image_b = next(unsupervised_dataloader)['image'].to(device, non_blocking=True)
                with torch.no_grad():
                    ema_pred_a = ema_model(unsupervised_image_a)[-1]
                    ema_pred_a = torch.nn.functional.interpolate(ema_pred_a, unsupervised_image_a.shape[2:4],
                                                                 mode='bilinear', align_corners=False)
                    ema_pred_b = ema_model(unsupervised_image_b)[-1]
                    ema_pred_b = torch.nn.functional.interpolate(ema_pred_b, unsupervised_image_a.shape[2:4],
                                                                 mode='bilinear', align_corners=False)

                    mask = cowmix.generate_cowmix_masks_like(
                        unsupervised_image_a,
                        mask_proportion_range=config['train']['mask_proportion_range'],
                        sigma_range=config['train']['sigma_range'])

                    mixed_ema_pred = cowmix.mix_with_mask(ema_pred_a, ema_pred_b, mask)

                    mixed_unsupervised_images = cowmix.mix_with_mask(unsupervised_image_a,
                                                                     unsupervised_image_b,
                                                                     mask)
                    del mask

                model.eval()
                mixed_student_pred = model(mixed_unsupervised_images)[-1]
                model.train()
                mixed_student_pred = torch.nn.functional.interpolate(mixed_student_pred, mixed_unsupervised_images.shape[2:4],
                                                                     mode='bilinear', align_corners=False)
                del mixed_unsupervised_images
                mixed_ema_pred_prob = torch.softmax(mixed_ema_pred, dim=1)
                confidence_modulator = (mixed_ema_pred_prob.max(dim=1).values > config['train']['confidence_threshold'])\
                    .to(mixed_ema_pred_prob)

                consistency_loss = torch.pow(torch.softmax(mixed_student_pred, dim=1) - mixed_ema_pred_prob,
                                             exponent=2.0)
                consistency_loss = (consistency_loss.sum(dim=1) * confidence_modulator).sum() / confidence_modulator.sum()

                consistency_loss = consistency_loss.mean()
                confidence_modulator = confidence_modulator.mean()
                reduced_confidence_modulator = utils.reduce_tensor(confidence_modulator) / world_size
                avg_confidence_modulator.update(reduced_confidence_modulator.item())


                unsup_loss = consistency_loss * config['train']['consistency_loss_weight'] * float(epoch > 25)
                reduced_unsup_loss = utils.reduce_tensor(unsup_loss) / world_size
                avg_unsupervised_loss.update(reduced_unsup_loss.item())
                unsup_loss.backward()
            else:
                avg_confidence_modulator.update(0)
                reduced_unsup_loss = torch.tensor(0, dtype=torch.float32)
                avg_unsupervised_loss.update(0)

            if step % config['train']['virtual_batch_size_multiplier'] == 0 and step != 0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), config['train']['gradient_clip_value'])
                optimizer.step()
                optimizer.zero_grad()

            if False:
                del mixed_ema_pred
                del unsup_loss

                mean_teacher.update_ema_variables(model, ema_model, alpha=config['train']['ema_model_alpha'])

            # Measure batch processing time
            batch_time.update(time.time() - tic)

            if step % config['train']['print_freq'] == 0 and rank == 0:
                msg = f'Epoch: {epoch} Step: {step} Batch time: {batch_time.average()} Loss: {avg_classification_loss.average()}'
                print(msg)
                summary_writer.add_scalar('train_classification_loss', reduced_classification_loss.item(), global_step)
                summary_writer.add_scalar('train_unsupervised_loss', reduced_unsup_loss.item(), global_step)

        if rank == 0:
            summary_writer.add_scalar('batch_time', batch_time.average(), global_step)
            summary_writer.add_scalar('train_loss_avg', avg_supervised_loss.average() + avg_unsupervised_loss.average(), global_step)
            summary_writer.add_scalar('train_supervised_loss_avg', avg_supervised_loss.average(), global_step)
            summary_writer.add_scalar('train_unsupervised_loss_avg', avg_unsupervised_loss.average(), global_step)
            summary_writer.add_scalar('train_classification_loss', avg_classification_loss.average(), global_step)
            summary_writer.add_scalar('train_confidence_modulator', avg_confidence_modulator.average(), global_step)


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

        for sample in tqdm(dataloader, smoothing=0.):
            image = sample['image'].to(device)
            mask = sample['semantic_mask'].to(device)

            features, pred_maps = model(image)

            # loss = config['train']['loss'](pred_maps, losses.smooth_labels(mask, alpha=0.05))
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
