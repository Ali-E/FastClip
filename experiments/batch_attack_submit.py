import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, help='dataset')
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('--method', default='clip', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='all')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--attack', default='fgsm')
parser.add_argument('--seed', default=-1, type=int)
args = parser.parse_args()


if __name__ == '__main__':
    base_classifier = args.base_classifier
    dataset = args.dataset
    attack = args.attack
    seed = args.seed    
    model = args.model

    if seed == -1:
        seed_list = [10**i for i in range(10)]#[:5]
    else:
        seed_list = [seed]

    steps = 50

    method = args.method
    if method == 'all':
        methods = ['lip4conv', 'nsedghi', 'gouk', 'miyato', 'orig', 'fastclip']
    elif method == 'clip':
        methods = ['lip4conv', 'nsedghi', 'gouk', 'miyato', 'fastclip']
    elif method == 'fast':
        methods = ['fastclip']
    else:
        methods = [method]

    mode = args.mode
    if mode == 'all':
        modes = ['wBN', 'noBN', 'clipBN_hard']
    elif mode == 'regular':
        modes = ['wBN', 'noBN']
    elif mode == 'dla':
        modes = ['wBN', 'clipBN_hard']
    else:
        modes = [mode]


    for mode in modes:
        for method in methods:
            if method == 'orig' and (mode == 'clipBN_hard'):
                continue
            for seed in seed_list:
                try:
                    # command = f"sbatch attack_job_submit.slurm {dataset} {model} {method} {mode} {base_classifier} {attack} {seed}"
                    command = f"python whitebox_multi.py {dataset} {model} {method} {mode} {base_classifier} {attack} --seed {seed}"
                    print(command)
                    os.system(command)

                except Exception as e:
                    print(e)
                    continue
