import logging
import os
import shutil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import config
import train
import utils.utils as utils


def distributed_train(rank, cfg_path):
    cfg = config.fromfile(cfg_path)
    if not cfg['common']['use_cpu']:
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=cfg['common']['world_size'])
        device_id = rank % cfg['common']['workers']
        device = f'cuda:{device_id}'
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
    else:
        dist.init_process_group("gloo", rank=rank, world_size=cfg['common']['world_size'])
        device = 'cpu'

    model = cfg['model']['model_fn'](cfg['common']['num_classes'])
    model = model.to(device)
    if device != 'cpu':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    device_ids = [device] if device != 'cpu' else None
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=True)

    discriminator = cfg['model']['discriminator'](in_channels=cfg['common']['num_classes'] + 3) # Num channels + RGB
    discriminator = discriminator.to(device)
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=device_ids,
                                                              find_unused_parameters=True)

    trainable_params = utils.get_trainable_params(model)
    # trainable_params = model.module.get_params_with_layerwise_lr(model.module, cfg['train']['base_lr'])
    # trainable_params = model.module.get_params_with_layerwise_lr(
    #     encoder_lr=cfg['train']['base_lr'] * 0.01,
    #     decoder_lr=cfg['train']['base_lr'],
    #     classifier_lr=cfg['train']['base_lr']
    # )

    optimizer = cfg['train']['optimizer'](params=trainable_params)

    discriminator_trainable_params = discriminator.module.parameters()
    discriminator_optimizer = cfg['train']['discriminator_optimizer'](params=discriminator_trainable_params)

    lr_scheduler = cfg['train']['lr_scheduler'](optimizer)
    discriminator_lr_scheduler = cfg['train']['lr_scheduler'](discriminator_optimizer)

    output_dir = cfg['common']['output_dir']
    train_dir = output_dir

    last_epoch = 0
    best_metric = None

    latest_checkpoint_path = os.path.join(train_dir, 'checkpoint.pth')
    if os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(latest_checkpoint_path,
                                map_location=lambda storage, loc: storage)
        best_metric = checkpoint['best_metric']
        last_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        discriminator.module.load_state_dict(checkpoint['discriminator_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
        print(f'Loaded checkpoint: Epoch {checkpoint["epoch"]}')
        del checkpoint
        torch.cuda.empty_cache()

    # Preaparing dataloaders here
    train_dataset = cfg['train']['dataset']()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg['train']['batch_size_per_worker'],
                                                   sampler=train_sampler,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   num_workers=cfg['train']['num_dataloader_workers'])

    unsupervised_dataset = cfg['train']['unsupervised_dataset']()
    unsupervised_sampler = torch.utils.data.distributed.DistributedSampler(unsupervised_dataset)
    unsupervised_dataloader = torch.utils.data.DataLoader(unsupervised_dataset,
                                                   batch_size=cfg['train']['batch_size_per_worker'],
                                                   sampler=unsupervised_sampler,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   num_workers=cfg['train']['num_dataloader_workers'])
    unsupervised_dataloader = iter(unsupervised_dataloader)

    val_dataset = cfg['val']['dataset']()
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg['val']['batch_size_per_worker'],
                                                 sampler=val_sampler,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 num_workers=cfg['val']['num_dataloader_workers'])

    if rank == 0:
        global_step = utils.calc_global_step(dataset_len=len(train_dataset), world_size=cfg['common']['world_size'],
                                             batch_size_per_worker=cfg['train']['batch_size_per_worker'],
                                             epoch=last_epoch)
        summary_writer = SummaryWriter(train_dir, purge_step=global_step, flush_secs=30)
    else:
        summary_writer = None

    torch.cuda.empty_cache()
    for epoch in range(last_epoch, 1000):
        train_sampler.set_epoch(epoch)
        global_step = utils.calc_global_step(dataset_len=len(train_dataset), world_size=cfg['common']['world_size'],
                                             batch_size_per_worker=cfg['train']['batch_size_per_worker'], epoch=epoch)

        if rank == 0:
            summary_writer.add_scalar('max_lr', utils.get_max_lr(optimizer), global_step=global_step)

        # Train one epoch
        train.train(model=model,
                    discriminator=discriminator,
                    optimizer=optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    dataloader=train_dataloader,
                    unsupervised_dataloader=unsupervised_dataloader,
                    epoch=epoch,
                    initial_step=global_step,
                    summary_writer=summary_writer,
                    config=cfg,
                    device=device)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

        # Validate model
        global_step = utils.calc_global_step(dataset_len=len(train_dataset), world_size=cfg['common']['world_size'],
                                             batch_size_per_worker=cfg['train']['batch_size_per_worker'],
                                             epoch=epoch + 1)
        val_loss = train.validate(model=model,
                                  dataloader=val_dataloader,
                                  epoch=epoch,
                                  initial_step=global_step,
                                  summary_writer=summary_writer,
                                  config=cfg,
                                  device=device)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

        lr_scheduler.step(val_loss, epoch=epoch + 1)
        discriminator_lr_scheduler.step(val_loss, epoch=epoch + 1)

        # TODO: Save checkpoint here
        if rank == 0:
            if best_metric is None or val_loss < best_metric:
                best_metric = val_loss
                torch.save(model.module.state_dict(),
                           os.path.join(train_dir, 'best.pth'))

            torch.save({
                'epoch': epoch + 1,
                'best_metric': best_metric,
                'state_dict': model.module.state_dict(),
                'discriminator_state_dict': discriminator.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'discriminator_optimizer': discriminator_optimizer.state_dict()
            }, os.path.join(train_dir, 'checkpoint.pth'))
            print(f'Saved checkpoint')

        torch.distributed.barrier()
        # TODO: Load checkpoint here

        if utils.get_max_lr(optimizer) <= cfg['train']['min_lr']:
            break


def cleanup():
    dist.destroy_process_group()


def main():
    os.environ['MASTER_ADDR'] = '192.168.1.73'
    os.environ['MASTER_PORT'] = '15000'
    cfg_path = 'configs/default_config.py'
    cfg = config.fromfile(cfg_path)
    os.makedirs(cfg['common']['output_dir'], exist_ok=True)
    shutil.copy2(cfg_path, cfg['common']['output_dir'])
    mp.spawn(distributed_train,
             args=(cfg_path,),
             nprocs=cfg['common']['world_size'],
             join=True)


if __name__ == '__main__':
    main()
