import os
from yaml import load,load_all,dump_all,Loader, Dumper
from easydict import EasyDict
from argparse import ArgumentParser
if __name__ == '__main__':
    f = open("test.yaml","r")
    config = load(f,Loader=Loader)
    experiment = EasyDict(config)
    parser = ArgumentParser("Experiment argument parser")
    parser.add_argument("--task_id",type=int, default=1)
    parser.add_argument("--online",action="store_true")
    args = parser.parse_args()
    command = f"python -u train.py --epochs {experiment.epochs} --project_name {experiment.project_name}  --batch_size {experiment.batch_size} --learning_rate {experiment.learning_rate} --task_id {args.task_id}"
    if args.online:
        command += " --online"
    os.system(command)
