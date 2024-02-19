import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='cifar', help='dataset')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--method', default='all', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='all')
parser.add_argument('--seed', default=-1, type=int)
args = parser.parse_args()


if __name__ == '__main__':
    dataset = args.dataset
    model = args.model
    if args.seed == -1:
        seed_list = [10**i for i in range(10)]
        # seed_list = [10**i for i in range(5)]
    else:
        seed_list = [args.seed]

    steps = 50 # this is clipBN steps if activated

    method = args.method
    if method == 'all':
        methods = ['lip4conv', 'gouk', 'nsedghi', 'miyato', 'orig', 'fastclip']
    elif method == 'clip':
        methods = ['lip4conv', 'nsedghi', 'gouk', 'miyato', 'fastclip']
    elif method[:4] == 'fast':
        methods = ['fastclip']
    else:
        methods = [method]

    if args.model  not in ['ResNet18', 'DLA']:
        raise ValueError('model must be either ResNet18 or DLA')

    mode = args.mode
    if mode == 'all':
        modes = ['wBN', 'noBN', 'clipBN_hard']
    elif mode == 'regular':
        modes = ['wBN', 'noBN']
    elif mode == 'clipBN':
        modes = ['clipBN_hard']
    elif mode == 'BN':
        modes = ['wBN','clipBN_hard']
    else:
        modes = [mode]

    for mode in modes:
        for method in methods:
            if method == 'orig' and (mode == 'clipBN_hard'):
                continue
            for seed in seed_list:
                try:
                    # command = f"sbatch job_submit.slurm {dataset} {method} {mode} {seed} {steps} {model}"
                    command = f"python main_jobsubmit.py --dataset {dataset} --method {method} --mode {mode} --seed {seed} --steps {steps} --model {model}"
                    os.system(command)

                except Exception as e:
                    print(e)
                    continue
