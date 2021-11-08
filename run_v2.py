import os
from yaml import load,load_all,dump_all,Loader, Dumper
from easydict import EasyDict
from argparse import ArgumentParser
if __name__ == '__main__':
    parser = ArgumentParser("Experiment argument parser")
    parser.add_argument("--task_id",type=int, default=1)
    parser.add_argument("--config",type=str,default="test")
    parser.add_argument("--online",action="store_true")
    args = parser.parse_args()
    f = open(f"config/{args.config}.yaml", "r")
    config = EasyDict(load(f, Loader=Loader))

    command = f"python -u train.py --epochs {config.epochs} --project_name {config.project_name}  --batch_size {config.batch_size} --learning_rate {config.learning_rate} --task_id {args.task_id}"
    if args.online:
        command += " --online"
    os.system(command)


