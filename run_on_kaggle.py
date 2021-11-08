import os

if __name__ == '__main__':
    os.system("git clone https://github.com/Pauljanson002/IncLearn.git")
    os.system("wandb login 5c39896b218f3aa477a66b1bd59cb4500e16a396")
    os.chdir("./IncLearn")
    print(os.getcwd())
    os.system("pip install -r requirements.txt")
    os.system("python -u run_v2.py --task_id 1 --config test")
