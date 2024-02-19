import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import pandas as pd
import random

import sys
sys.path.append('../')
from models import *
from others.datasets import get_dataset

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack, LinfMomentumIterativeAttack, \
    CarliniWagnerL2Attack, JacobianSaliencyMapAttack, ElasticNetL1Attack
from advertorch.attacks.utils import attack_whole_dataset

import matplotlib
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Whitebox attack')
parser.add_argument('dataset', type=str)
parser.add_argument('model', type=str)
parser.add_argument('method', type=str)
parser.add_argument('mode', type=str)
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("attack_type", type=str, help="choose from [fgsm, pgd, mim, bim, jsma, cw, ela, autoattack]")
parser.add_argument('--seed', default=1, type=float)
parser.add_argument('--adv-eps', default=0.02, type=float)
parser.add_argument('--adv-steps', default=50, type=int)
parser.add_argument('--random-start', default=1, type=int)
parser.add_argument('--coeff', default=0.02, type=float) # for jsma, cw, ela

args = parser.parse_args()


def main(model, device='cpu', clip_flag=True, clip_opt_iter=1, init_delay=0, summary=False):

    if args.dataset == 'cifar':
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        testset = torchvision.datasets.CIFAR10( root='./../data_new', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    elif args.dataset == 'mnist':
        testset = get_dataset('mnist', 'test')
        testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=128, num_workers=2)

    else:
        print('not the correct dataset!')
        exit(0)


    loss_fn = nn.CrossEntropyLoss()

    correct_or_not = []
    for i in range(args.random_start):
        print("Phase %d" % (i))
        torch.manual_seed(i)
        torch.cuda.manual_seed_all(i)
        np.random.seed(i)
        random.seed(i)

        test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

        if (args.attack_type == "pgd"):
            adversary = LinfPGDAttack(
                model, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
                # nb_iter=args.adv_steps, eps_iter=args.adv_eps / 5, rand_init=True, clip_min=0., clip_max=1.,
                targeted=False)
        elif (args.attack_type == "fgsm"):
            adversary = GradientSignAttack(
                model, loss_fn=loss_fn, eps=args.adv_eps,
                clip_min=0., clip_max=1., targeted=False)
        elif (args.attack_type == "mim"):
            adversary = LinfMomentumIterativeAttack(
                model, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, clip_min=0., clip_max=1.,
                targeted=False)
        elif (args.attack_type == "bim"):
            adversary = LinfBasicIterativeAttack(
                model, loss_fn=loss_fn, eps=args.adv_eps,
                nb_iter=args.adv_steps, eps_iter=args.adv_eps / 10, clip_min=0., clip_max=1.,
                targeted=False)
        elif (args.attack_type == "cw"):
            adversary = CarliniWagnerL2Attack(
                model, confidence=0.1, max_iterations=1000, clip_min=0., clip_max=1.,
                targeted=False, num_classes=10, binary_search_steps=1, initial_const=args.coeff)


        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")

        correct_or_not.append(label == advpred)
            
    correct_or_not = torch.stack(correct_or_not, dim=-1).all(dim=-1)

    print("")
    if (args.attack_type == "cw" or args.attack_type == "ela"):
        print("%s (c = %.2f): %.2f (Before), %.2f (After)" % (args.attack_type.upper(), args.adv_eps,
            100. * (label == pred).sum().item() / len(label),
            100. * correct_or_not.sum().item() / len(label)))
    elif (args.attack_type == "jsma"):
        print("%s (gamma = %.2f): %.2f (Before), %.2f (After)" % (args.attack_type.upper(), args.adv_eps,
            100. * (label == pred).sum().item() / len(label),
            100. * correct_or_not.sum().item() / len(label)))
    else:
        print("%s (eps = %.2f): %.2f (Before), %.2f (After)" % (args.attack_type.upper(), args.adv_eps,
            100. * (label == pred).sum().item() / len(label),
            100. * correct_or_not.sum().item() / len(label)))

    return 100. * (label == pred).sum().item() / len(label), 100. * correct_or_not.sum().item() / len(label)


if __name__ == '__main__':
    print('device: ', device)
    if args.dataset == 'cifar':
        in_chan = 3
    else:
        in_chan = 1
    global_clip_flag=False
    name = '/ckpt_best_test.pth'
    # name = '/ckpt.pth'
    seed_list = [10**i for i in range(10)][:5]

    seed = args.seed
    if seed == -1:
        # seed_list = [1, 10, 100, 1000, 10000]
        seed_list = [10**i for i in range(10)]#[:5]
    else:
        seed_list = [seed]

    method = args.method
        
    concat_sv = False
    clip_outer_flag = False
    if method == 'catclip':
        concat_sv = True
        clip_outer_flag = True

    bn_flag = True
    bn_clip_flag = False
    mode = args.mode
    if mode not in  ['wBN', 'noBN', 'clipBN_hard', 'clipBN_soft']:
        print('mode not supported')
        exit(0)
    if mode == 'wBN':
        mode = ''

    if mode == 'noBN':
        bn_flag = False

    if mode in  ['clipBN_hard', 'clipBN_soft']:
        bn_clip_flag = True


    acc_list = []
    adv_acc_list = []
    for seed in seed_list:
        try:
            base_classifier = args.base_classifier + method + '_' + mode + '_' + str(int(seed)) + name
            # base_classifier = args.base_classifier + str(seed) + name
            print(base_classifier)
            outdir = '/'.join(base_classifier.split('/')[:-1])
            checkpoint = torch.load(base_classifier)

            if args.model == 'ResNet18':
                bn_hard = False
                steps_count = 50

                if method == 'lip4conv':
                    net = ResNet18_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip_flag, clip_linear=True, device=device)
                elif method in ['nsedghi', 'gouk', 'orig']:
                    net = ResNet18_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip_flag, clip_linear=False, device=device)
                elif method == 'miyato':
                    net = ResNet18_miyato(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip_flag, device=device)
                elif method[:4] == 'fast' or method == 'catclip':
                    net = ResNet18(concat_sv=concat_sv, in_chan=in_chan, device=device, clip=1., clip_concat=1., clip_flag=False, bn=bn_flag, bn_clip=bn_clip_flag, bn_hard=bn_hard, clip_steps=50, bn_count=50, clip_outer=clip_outer_flag, clip_opt_iter=1, summary=True, save_info=False, outer_iters=1, outer_steps=100)

                    # net = ResNet18(in_chan=in_chan, device=device, clip=1., clip_flag=False, bn=bn_flag, bn_clip=bn_clip_flag, clip_steps=steps, clip_outer=False, clip_opt_iter=1, summary=True, save_info=False)

                    # net = ResNet18(concat_sv=True, in_chan=in_chan, device=device, clip=1., clip_flag=bn_clip_flag, bn=bn_flag, bn_clip=False, clip_steps=steps, clip_outer=False, clip_opt_iter=1, summary=True, save_info=False)

            elif args.model == 'DLA':
                bn_clip = bn_clip_flag
                bn_hard = False
                steps_count = 50
                
                if method == 'lip4conv':
                    net = DLA_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_linear=True, bn_count=steps_count, device=device)
                elif method in ['nsedghi', 'gouk', 'orig']:
                    net = DLA_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_linear=False, bn_count=steps_count, device=device)
                    # net = ResNet18_asl(in_chan=in_chan, bn=True, bn_clip=False, clip_linear=False, device=device)
                elif method == 'miyato':
                    net = DLA_miyato(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=steps_count, device=device)
                elif method[:4] == 'fast' or method == 'catclip':
                    net = DLA(concat_sv=concat_sv, in_chan=in_chan, device=device, clip=1., clip_concat=1., clip_flag=False, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_steps=50, bn_count=50, clip_outer=clip_outer_flag, clip_opt_iter=1, summary=True, writer=None)


            model = net.to(device)
            model.load_state_dict(checkpoint['net'], strict=False)
            # model.load_state_dict(checkpoint['net'])#, strict=False)
            model.eval()

            print ('Model loaded')

            acc, adv_acc = main(model, device=device, clip_flag=global_clip_flag, clip_opt_iter=1, summary=True, init_delay=0)
            acc_list.append(acc)
            adv_acc_list.append(adv_acc)
        except Exception as e:
            print(e)
            continue

    print('acc list: ', acc_list)
    print(np.mean(acc_list), np.std(acc_list))

    print('adv acc list: ', adv_acc_list)
    print(np.mean(adv_acc_list), np.std(adv_acc_list))

    df = pd.DataFrame({'acc': acc_list, 'adv': adv_acc_list})
    # df.to_csv('paper_results_' + args.dataset  + '/' + base_classifier.split('/')[3] + '_' + str(args.adv_eps) + '_' + args.attack_type + '.csv', index=False)

    df.to_csv(outdir + '/' + args.attack_type + '_' + str(args.adv_eps) + '_' + str(args.coeff) +  '.csv', index=False)
    # df.to_csv(outdir + '/' + args.attack_type + '_' + str(args.adv_eps) + '_' + str(args.coeff) +  '_last.csv', index=False)



