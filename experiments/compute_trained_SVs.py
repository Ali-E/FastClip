import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import sys
import os
import pandas as pd

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)

sys.path.append('../')
from models import *
from others.datasets import get_dataset

import matplotlib
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='computing SVs in a trained model')
parser.add_argument('dataset', type=str)
parser.add_argument('method', type=str)
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('--direct', default=False)
parser.add_argument('--seed', default=-1, type=int)
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')

args = parser.parse_args()


def main(model, width_list, device='cpu', n_iters=1000, miyato=False):
    idx = 0
    sv_list = []
    # print(model)
    for (m_name, m) in model.named_modules():
        # print(m_name, m)
        # if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        #     if isinstance(m, torch.nn.Linear):
        #         x0 = torch.randn(1, m.weight.shape[1], device=device)
        #         const = 'Linear'
        if isinstance(m, torch.nn.Conv2d):
            # print('Conv2d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            if miyato:
                input_channel = m.weight_bar.shape[1]
            else:
                input_channel = m.weight.shape[1]
            if args.dataset == 'cifar':
                VT_shape = [1, input_channel, width_list[idx], width_list[idx]]
            else:
                VT_shape = [1, input_channel, width_list[idx], width_list[idx]]
            x0 = torch.randn(VT_shape, device=device)
            const = 'Conv2d_' + str(idx)
            idx += 1

            qr = power_qr(lambda x: m(x) - m(torch.zeros_like(x)), x0.clone().detach(), n_iters=n_iters, device=device)
            sv_list.append((qr[-1][0], const))

    return sv_list


def get_avg_std_sv(sv_seed_list):
    sv_std_list = []
    sv_avg_list = []
    for i in range(len(sv_seed_list[0])):
        sv_list = []
        for j in range(len(sv_seed_list)):
            sv_list.append(sv_seed_list[j][i][0].item())
        sv_std = np.std(sv_list)
        sv_avg = np.mean(sv_list)
        sv_std_list.append((sv_std, sv_seed_list[0][i][1]))
        sv_avg_list.append((sv_avg, sv_seed_list[0][i][1]))
    return sv_avg_list, sv_std_list


if __name__ == '__main__':
    
    if args.dataset == 'cifar':
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        testset = torchvision.datasets.CIFAR10( root='./../data_new', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

    elif args.dataset == 'mnist':
        testset = get_dataset('mnist', 'test')
        testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=128, num_workers=1)


    print('device: ', device)
    if args.dataset == 'cifar':
        # width_list = [32] + [32,32,32,32,32,16,16,16,16,16,8,8,8,8,8,4,4,4,4,4] ### ResNet18
        in_chan = 3
        if args.model == 'ResNet18':
            width_list = [32] + [32,32,32,32,32,16,16,16,16,16,8,8,8,8,8,4,4,4,4,4] ### ResNet18
        elif args.model == 'DLA':
            width_list = [32] + [32 for i in range(9)] + [16 for i in range(14)] + [8 for i in range(14)] + [4 for i in range(6)] ### DLA
        elif args.model == 'VGG19':
            vgg_flag = True
            width_list = [32, 32, 16, 16, 8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2]
            # width_list = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
    else:
        # width_list = [28] + [28,28,28,28,28,14,14,14,14,14,7,7,7,7,7,4,4,4,4,4] ### ResNet18
        in_chan = 1
        if args.model == 'ResNet18':
            width_list = [28] + [28,28,28,28,28,14,14,14,14,14,7,7,7,7,7,4,4,4,4,4] ### ResNet18
        elif args.model == 'DLA':
            width_list = [28] + [28 for i in range(9)] + [14 for i in range(14)] + [7 for i in range(14)] + [4 for i in range(6)] ### DLA


    global_clip_flag=False
    name = '/ckpt_best_test.pth'
    # name = '/ckpt.pth'
    seed_list = [10**i for i in range(5)][:5]
    # seed_list = [10**i for i in range(5)][:3]
    if args.seed != -1:
        seed_list = [args.seed] 

    n_iters = 2000


    method = args.method
    if method == 'all':
        methods = ['miyato', 'lip4conv', 'nsedghi', 'gouk', 'orig', 'fastclip']
    elif method[:4] == 'fast':
        methods = ['fastclip']
    else:
        methods = [method]

    # modes = ['', 'noBN', 'clipBN_hard', 'clipBN_soft']
    modes = ['']

    data_rows = []
    if args.direct:
        seed_list = [0] 
    for mode in modes:
        for method in methods:
            print(method, mode)
            if method == 'orig' and (mode == 'clipBN_hard'):
                continue
            sv_seed_list = []
            acc_list = []
            for seed in seed_list:
                miyato_flag = False 
                # try:
                if True:
                    if args.direct:
                        base_classifier = args.base_classifier + name
                    else:
                        base_classifier = args.base_classifier + method + '_' + mode + '_' + str(seed) + name

                    print(base_classifier)
                    checkpoint = torch.load(base_classifier)

                    if args.model == 'ResNet18':
                        if method == 'lip4conv':
                            net = ResNet18_orig(in_chan=in_chan, bn=True, bn_clip=False, clip_linear=True, device=device)
                        elif method in ['nsedghi', 'gouk', 'orig']:
                            net = ResNet18_orig(in_chan=in_chan, bn=True, bn_clip=False, clip_linear=False, device=device)
                        elif method == 'miyato':
                            miyato_flag = True
                            net = ResNet18_miyato(in_chan=in_chan, bn=True, bn_clip=False, device=device, eval_mode=True)
                        elif method[:4] == 'fast':
                            net = ResNet18(in_chan=in_chan, device=device, clip=1., clip_flag=False, bn=True, bn_clip=False, clip_steps=100, clip_outer=False, clip_opt_iter=1, summary=True, save_info=False)


                    elif args.model == 'DLA':
                        bn_flag = True
                        bn_clip = False
                        bn_hard = False
                        steps_count = 100
                        if method == 'lip4conv':
                            net = DLA_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_linear=True, bn_count=steps_count, device=device)
                        elif method in ['nsedghi', 'gouk', 'orig']:
                            net = DLA_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_linear=False, bn_count=steps_count, device=device)
                        elif method == 'miyato':
                            net = DLA_miyato(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=steps_count, device=device, eval_mode=True)
                        elif method[:4] == 'fast':
                            net = DLA(in_chan=in_chan, device=device, clip=1., clip_flag=True, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_steps=100, init_delay=0, bn_count=steps_count, clip_outer=False, clip_opt_iter=1, summary=True, writer=None)
            

                    model = net.to(device)
                    model.load_state_dict(checkpoint['net'], strict=False)
                    model.eval()

                    print ('Model loaded')

                    sv_list = main(model, width_list, device=device, n_iters=n_iters, miyato=miyato_flag)
                    sv_seed_list.append(sv_list)

                    model.eval()
                    test_loss = 0
                    correct = 0
                    total = 0
                    batch_idx = -1
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(testloader):
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs = net(inputs)
                            # loss = criterion(outputs, targets)
                            # test_loss += loss.item()
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()

                    test_acc = 100.*correct/total
                    print('acc: ', test_acc)
                    acc_list.append(test_acc)

                # except Exception as e:
                #     print(e)
                #     continue

            if len(sv_seed_list) > 0:
                sv_avg, sv_std = get_avg_std_sv(sv_seed_list)
                test_acc = np.mean(acc_list)

            new_row = [mode, method, test_acc] + [sv_avg[i][0] for i in range(len(sv_avg))] + [sv_std[i][0] for i in range(len(sv_std))]
            data_rows.append(new_row)
            model = None
            net = None
            

    df = pd.DataFrame(data_rows, columns=['mode', 'method', 'test_acc'] + [sv_avg[i][1] + '_avg' for i in range(len(sv_avg))] + [sv_std[i][1] + '_std' for i in range(len(sv_std))])
    # f = open('paper_table_' + args.dataset  + '/' + base_classifier.split('/')[3] + '_' + str(args.adv_eps) + '_' + args.attack_type + '.csv', 'a')

    # f = open('paper_sv_table_' + args.dataset  + '/' + base_classifier.replace("/", "_") + '_' + '.csv', 'a')
    f = open(args.base_classifier + '_layerwise_SV_' + name[1:-4] + '.csv', 'a')

    f.write('# ' + base_classifier + '\n')
    df.to_csv(f, float_format='%.2f')



