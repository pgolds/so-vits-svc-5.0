import os
import argparse
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from vits_extend.train import train

torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the model for logging, saving checkpoint")
    parser.add_argument('--pth_dir', type=str, required=True,
                        help="saving checkpoint path")
    parser.add_argument('-d', '--dataRoot', type=str, required=True,
                        help="data root filelists")
    parser.add_argument('-s', '--save_interval', type=int, default=1500,
                        help="saving step checkpoint")
    parser.add_argument('-m', '--max_step', type=int, default=1510,
                        help="saving step checkpoint")
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    # 替换
    training_files = os.path.join(args.dataRoot, "filelists", "train.txt")
    validation_files = os.path.join(args.dataRoot, "filelists", "valid.txt")
    hp.data.training_files = training_files
    hp.data.validation_files = validation_files

    if args.pth_dir is not None:
        hp.log.pth_dir = args.pth_dir
    hp.log.save_interval = args.save_interval
    hp.log.max_step = args.max_step

    assert hp.data.hop_length == 320, \
        'hp.data.hop_length must be equal to 320, got %d' % hp.data.hop_length

    args.num_gpus = 0
    torch.manual_seed(hp.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.train.seed)
        args.num_gpus = torch.cuda.device_count()
        print('Batch size per GPU :', hp.train.batch_size)

        if args.num_gpus > 1:
            mp.spawn(train, nprocs=args.num_gpus,
                     args=(args, args.checkpoint_path, hp, hp_str,))
        else:
            train(0, args, args.checkpoint_path, hp, hp_str)
    else:
        print('No GPU find!')
