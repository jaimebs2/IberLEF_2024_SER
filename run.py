from audio_language_model.train import train
from audio_language_model.predict import predict
import json
import argparse

if __name__ == "__main__":
    with open('audio_language_model/config.json') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=config['fold'])
    parser.add_argument('--batch', type=int, default=config['batch_size'])
    parser.add_argument('--lr', type=float, default=config['lr'])
    parser.add_argument('--gender', type=int, default=config['gender'])
    parser.add_argument('--model', type=str, default=config['model'])
    parser.add_argument('--phones', type=int, default=config['phones'])
    parser.add_argument('--ckpt', type=str, default=config['ckpt'])
    parser.add_argument('--lr_scheduler', type=str, default=config['lr_scheduler'])
    parser.add_argument('--accum_batch', type=int, default=config['accumulate_grad_batches'])
    parser.add_argument('--train_decoder', type=int, default=0)
    args = parser.parse_args()

    config['batch_size'] = args.batch
    config['lr'] = args.lr
    config['fold'] = args.fold
    config['model'] = args.model
    config['lr_scheduler'] = args.lr_scheduler
    config['accumulate_grad_batches'] = args.accum_batch
    if args.gender == 0:
        config['gender'] = False
    else:
        config['gender'] = True
    if args.phones == 0:
        config['phones'] = False
    else:
        config['phones'] = True
    if args.train_decoder == 0:
        config['trainable_layers'] = ["audio_encoder", "projector"]
    config['ckpt'] = args.ckpt

    if True:#not config['predict']:
        train(config=config,
            debbug=False
            )
    else:
        preds = predict(config=config)
        print(preds)