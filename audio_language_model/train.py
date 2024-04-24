import torch
from src.data_module import EmoSPeechDataModule, EmoSPeechDataset, collate_fn_predict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import random
import string
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
import os
from audio_language_model.alm import WhisperGemmaClassifier
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from audio_language_model.freeze_callback import FreezeCallback
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm import tqdm
from torch.utils.data import DataLoader

def train(config, debbug:bool=False):

    torch.set_float32_matmul_precision('high')
    
    run_name = config['model'] + '_'+str(config['fold'])
    run_name = run_name+'_'+''.join(random.choices(string.ascii_uppercase, k=3))
    if config['model'].startswith('WG'):
        config['audio_encoder_name'] = 'whisper-large-v3'
    elif config['model'].startswith('E2VG'):
        config['audio_encoder_name'] = 'emotion2vec'

    seed_everything(42)

    model = WhisperGemmaClassifier(**config)
    db = EmoSPeechDataModule(**config)
    
    if config['train_val']:
        group='finalists'
    else:
        group=config['model']
    logger_kwargs = {'group': group}
    logger = WandbLogger(
        project="IberLEF_2024",
        name=run_name,
        log_model='False',
        **logger_kwargs,
        )

    if not config['train_val']:
        monitor = 'val/f1'
        save_last = False
        val_check_interval = 0.5
        enable_checkpointing = False
    else:
        monitor = 'train/f1'
        save_last = True
        val_check_interval = 0.0
        enable_checkpointing = True
    
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath='checkpoints/',
        filename=run_name,
        save_top_k=1,
        mode='max',
        save_last=save_last,
        )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    if config['deepspeed']:
        strategy = "deepspeed_stage_3"
    else:
        strategy = DDPStrategy(find_unused_parameters=True)

    freeze = FreezeCallback()
    early_stopping = EarlyStopping(monitor='val/f1', mode='max', min_delta=0.00, patience=5)

    if not config['train_val']:
        callbacks = [lr_monitor, freeze, early_stopping]
    else:
        callbacks = [lr_monitor, freeze, checkpoint_callback]

    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        min_epochs=1, 
        max_epochs=config['max_epochs'],
        fast_dev_run=debbug,
        logger=logger,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        val_check_interval=val_check_interval,
        callbacks=callbacks,
        precision="16-mixed",#"bf16",
        strategy = strategy,
        enable_checkpointing=enable_checkpointing,
        log_every_n_steps=20/config['batch_size'],
        )

    if not config['train_val']:
        trainer.fit(model, db)# ckpt_path='checkpoints/prueba_0_BBG.ckpt')
    else:
        db.setup()
        trainer.fit(model, train_dataloaders=db.train_dataloader())
        