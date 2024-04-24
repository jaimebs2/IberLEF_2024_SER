from pytorch_lightning.callbacks import Callback

class FreezeCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        if "audio_encoder" not in pl_module.trainable_layers:
            pl_module.model.audio_encoder.requires_grad_(False)
        pl_module.model.llm.requires_grad_(False)