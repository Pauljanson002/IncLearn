import os
if __name__ == '__main__':
    os.system("git commit -a -m \"Training \"")
    os.system("git push origin master")
    os.system("kaggle kernels push")