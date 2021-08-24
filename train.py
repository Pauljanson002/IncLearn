import argparse
import wandb
import torch
import os
import numpy as np
from Trainer import Trainer

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--project_name', type=str, default='project_null')
    # Todo add model as argument
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--task_id',type=int,default=1)

    # Wandb configs
    os.environ['WANDB_MODE'] = 'offline'
    args = parser.parse_args()
    if args.online:
        os.environ['WANDB_MODE'] = 'online'
    trainer = Trainer(args.epochs, args.learning_rate, args.batch_size, task_size=10, num_class=10)
    config = {
        'epochs': args.epochs,
        'project_name': args.project_name,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    run = wandb.init(
        project=args.project_name,
        reinit=True,
        config=config
    )
    wandb.run.name = f"task_id : {args.task_id}"
    wandb.watch(trainer.model)
    print(trainer.model)
    for i in range(1,args.task_id):
        trainer.beforeTrain()
        trainer.train(resume=True,task_id=i)
        trainer.afterTrain(task_id=i,no_save=True)
    trainer.beforeTrain()
    trainer.train(resume=False,task_id=args.task_id)
    trainer.afterTrain(task_id=args.task_id,no_save=False)
    run.finish()

