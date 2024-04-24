import torch
from src.data_module import EmoSPeechDataModule
from ALM.ALM import WhisperPhi
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import random
import string
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
import os
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from collections import OrderedDict

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

def model(config, train:bool=True, debbug:bool=False):

    torch.set_float32_matmul_precision('high')
    
    run_name = config['model'] + '_'+str(config['fold'])
    run_name = run_name+'_'+''.join(random.choices(string.ascii_uppercase, k=3))
    
    if config['ckpt']!="none":
        #model = WhisperPhi.load_from_checkpoint(config['ckpt'])
        WhisperPhi(**config)

        model = WhisperPhi.load_state_dict(config['ckpt'])
        #model.configure_model()
        '''
        state_dict = torch.load('/home/jaime/repos/IberLEF_2024_SER/checkpoints/wg_en.ckpt')
        state_dict['state_dict']['loss_fn.weight'] = torch.tensor([1,1,1,1,1,1])
        encoder_state_dict = {}
        for key, value in state_dict['state_dict'].items():
            if key.startswith("audio_encoder.model.encoder"):
                new_key = key.replace("audio_encoder.model.encoder.", "audio_encoder.encoder.")
                encoder_state_dict[new_key] = value
            elif key.startswith("audio_encoder.model.projector"):
                new_key = key.replace("audio_encoder.model.projector.", "projector.")
                encoder_state_dict[new_key] = value
            else:
                encoder_state_dict[key] = value
        state_dict['state_dict'] = encoder_state_dict
        torch.save(state_dict, "checkpoints/wg_en.ckpt")'''
       #del state_dict
    else:
        model = WhisperPhi(**config)

    seed_everything(42)

    '''
    for name, param in model.audio_encoder.model.named_parameters():
        if name.startswith(tuple(config['trainable_layers'])):
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, param in model.llm.named_parameters():
        if name.startswith(tuple(config['trainable_layers'])):
            param.requires_grad = True
        else:
            param.requires_grad = False
    '''
    db = EmoSPeechDataModule(**config)
    
    logger_kwargs = {'group': config['model']}
    logger = WandbLogger(
        project="IberLEF_2024",
        name=run_name,
        log_model='False',
        **logger_kwargs,
        )

    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath='checkpoints/',
        filename=run_name,
        save_top_k=1,
        mode='min',
        )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        min_epochs=1, 
        max_epochs=config['max_epochs'],
        fast_dev_run=debbug,
        logger=logger,
        accumulate_grad_batches=1,
        val_check_interval=0.5,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="16-mixed",#"bf16",
        #strategy = "deepspeed_stage_3",
        strategy = DDPStrategy(find_unused_parameters=False),
        enable_checkpointing=True,
        log_every_n_steps=20/config['batch_size'],
        )
    if train:
        trainer.fit(model, db) #ckpt_path=config['ckpt']
    else:
        db.setup()
        preds = trainer.predict(model, db.test_dataloader())
        p=[p[0].tolist() for p in preds]
        preds=[item for sublist in p for item in sublist]
        import pandas as pd
        preds_name = 'whispergemma_zeroshot_test' + run_name + '.csv'
        pd.DataFrame(preds).to_csv(preds_name, index=False)
