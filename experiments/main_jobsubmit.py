'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import argparse
import pandas as pd
import random
import time

import sys
sys.path.append('../')
from models import *
from others.datasets import get_dataset
import others.bounds as bounds

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='cifar', help='dataset')
parser.add_argument('--model', default='ResNet18', help='Deep Learning model to train')
parser.add_argument('--method', default='catclip', help='clipping method (use orig for no clipping)')
parser.add_argument('--mode', default='wBN', help='what to do with BN layers (leave empty for keeping it as it is)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--seed', default=1, type=int, help='seed value')
parser.add_argument('--steps', default=50, type=int, help='setp count for clipping BN')

parser.add_argument('--catsn', default=-1, type=float)
parser.add_argument('--convsn', default=1., type=float)
parser.add_argument('--outer_steps', default=100, type=int)
parser.add_argument('--convsteps', default=100, type=int)
parser.add_argument('--opt_iter', default=5, type=int)
parser.add_argument('--outer_iters', default=1, type=int)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==========', device)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    print('chosen: ', device)
    cudnn.benchmark = True

# Training
def train(epoch, optimizer, scheduler, criterion, writer=None, model_path="./checkpoints/", width_list=[None], new_sedghi=False, lip4conv=False, gouk_correct=False):
    print('\nEpoch: %d' % epoch)
    global count_setp
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = -1

    start = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if epoch == 0 and batch_idx == 0:
            print('inputs shape: ', inputs.shape)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        reg_loss_conv = torch.tensor([0.0], requires_grad=True).to(device=loss.device)
        
        ############################# lip4conv:
        if lip4conv:
            for (kernel, input_size) in net.get_all_kernels():
                bound = bounds.estimate(
                    kernel,
                    n=input_size[0],
                    name_func="ours_backward",
                    n_iter=6,
                )
                reg_loss_conv = reg_loss_conv + F.threshold(
                    bound, 1.0, 0.0
                )

            reg_loss_conv = 0.1 * reg_loss_conv
        #############################

        loss = loss + reg_loss_conv
        loss.backward()
        optimizer.step()

        ########################## Gouk:
        if gouk_correct and (count_setp > 0 or epoch != 0):# and count_setp % 100 == 0:

            with torch.no_grad():
                constrain_conv(
                    net,
                    mode='divide-by-largest',
                    conv_clip_assign_value=1,
                    linear_clip_assign_value=1,
                    iterations=5,
                    orthogonal=-1,
                    outputs=width_list,
                )
        #################

        ########################## Senderovich:
        if new_sedghi and (count_setp > 0 or epoch != 0) and count_setp % 100 == 0:

            with torch.no_grad():
                constrain_conv(
                    net,
                    mode='clip',
                    conv_clip_assign_value=1,
                    linear_clip_assign_value=1,
                    orthogonal=-1,
                    outputs=width_list,
                )
        #################


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        count_setp += 1

    tot_time = time.time() - start
    print('time: ', tot_time)

    writer.add_scalar('train/acc', 100.*correct/total, epoch)
    writer.add_scalar('train/loss', train_loss/(batch_idx+1), epoch)
    writer.add_scalar('train/time', tot_time, epoch)
    print('train/acc', 100.*correct/total)
    print('train/loss', train_loss/(batch_idx+1))

    scheduler.step()

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, model_path)

    return train_loss/(batch_idx+1), 100.*correct/total


def test(epoch, criterion, writer=None, model_path="./checkpoints/"):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = -1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc

        print('Saving Best..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'lr': scheduler.get_last_lr(),
        }
        torch.save(state, model_path)

    writer.add_scalar('test/acc', 100.*correct/total, epoch)
    writer.add_scalar('test/loss', test_loss/(batch_idx+1), epoch)
    mode = 'test'
    print(mode + '/acc', 100.*correct/total)
    print(mode + '/loss', test_loss/(batch_idx+1))

    return test_loss/(batch_idx+1), 100.*correct/total


if __name__ == "__main__":
    method = args.method
    steps_count = args.steps  #### BN clip steps for hard clip
    concat_sv = False
    clip_outer_flag = False
    outer_steps = args.outer_steps
    outer_iters = args.outer_iters
    if args.catsn > 0.:
        concat_sv = True
        clip_steps = args.convsteps
        clip_outer_flag = True

    mode = args.mode
    bn_flag = True
    bn_clip = False
    bn_hard = False
    opt_iter = args.opt_iter
    if mode == 'wBN':
        mode = ''
        bn_flag = True
        bn_clip = False
        clip_steps = 50
    elif mode == 'noBN':
        bn_flag = False
        bn_clip = False
        opt_iter = 1
        clip_steps = 100
    elif mode == 'clipBN_hard':
        bn_flag = True
        bn_clip = True
        bn_hard = True
        clip_steps = 100
    else:
        print('unknown mode!')
        exit(0)

    seed_in = args.seed
    for seed in [seed_in]:
        print('seed.....', seed)
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        count_setp = 0

        seed_val = seed
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        np.random.seed(seed_val)
        random.seed(seed_val)

        T_max = 200
        manual_SV = False

        clip_flag    = False
        new_sedghi   = False
        miyato_flag  = False
        gouk_correct = False
        lip4conv     = False
        orig_flag    = False

        if method[:4] == 'fast':
            clip_flag    = True
        elif method == 'catclip':
            clip_flag    = True
        elif method == 'nsedghi':
            new_sedghi   = True
        elif method == 'miyato':
            miyato_flag  = True
        elif method == 'gouk':
            gouk_correct = True
        elif method == 'lip4conv':
            lip4conv     = True
        elif method == 'orig':
            orig_flag    = True
        else:
            print('unknown method!')
            exit(0)

        if new_sedghi or gouk_correct:
            from others.utils_practical import *
            # from others.tt_dec_layer import ConvDecomposed2D_t, conv2tt

        # Data
        print('==> Preparing data..')
        if args.dataset == 'cifar':
            print('cifar!')
            in_chan = 3
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = torchvision.datasets.CIFAR10(
                root='./../data_new', train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=True, num_workers=1)

            testset = torchvision.datasets.CIFAR10(
                root='./../data_new', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=128, shuffle=False, num_workers=1)

        else:
            print('mnist!')
            in_chan = 1
            trainset = get_dataset('mnist', 'train')
            testset = get_dataset('mnist', 'test')
            trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=128, num_workers=1)
            testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=128, num_workers=1)


        outdir = "logs/" + args.dataset + "/" + args.model + "_models/method_" + method + "_" + mode + "_" + str(seed_val) + "/"

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        writer = SummaryWriter(outdir)

        print('==> Building model..')
        print('------------> outdir: ', outdir)

        if args.model == 'ResNet18':
            if lip4conv:
                net = ResNet18_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_linear=True, bn_count=steps_count, device=device)
            elif new_sedghi or gouk_correct or orig_flag:
                net = ResNet18_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_linear=False, bn_count=steps_count, device=device)
            elif miyato_flag:
                net = ResNet18_miyato(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=steps_count, device=device)
            elif clip_flag:
                net = ResNet18(concat_sv=concat_sv, in_chan=in_chan, device=device, clip=args.convsn, clip_concat=args.catsn, clip_flag=True, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_steps=clip_steps, bn_count=steps_count, clip_outer=clip_outer_flag, clip_opt_iter=opt_iter, summary=True, writer=writer, save_info=False, outer_iters=outer_iters, outer_steps=outer_steps)

        elif args.model == 'DLA':
            if lip4conv:
                net = DLA_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_linear=True, bn_count=steps_count, device=device)
            elif new_sedghi or gouk_correct or orig_flag:
                net = DLA_orig(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_linear=False, bn_count=steps_count, device=device)
            elif miyato_flag:
                net = DLA_miyato(in_chan=in_chan, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, bn_count=steps_count, device=device)
            elif clip_flag:
                net = DLA(concat_sv=concat_sv, in_chan=in_chan, device=device, clip=args.convsn, clip_concat=args.catsn, clip_flag=True, bn=bn_flag, bn_clip=bn_clip, bn_hard=bn_hard, clip_steps=clip_steps, init_delay=0, bn_count=steps_count, clip_outer=clip_outer_flag, clip_opt_iter=opt_iter, summary=True, writer=writer, outer_iters=outer_iters, outer_steps=outer_steps)

        net = net.to(device)
        criterion = nn.CrossEntropyLoss()

        width_list = [None]
        if args.dataset == 'cifar':
            if args.model == 'ResNet18':
                width_list = [32,32,32,32,32,16,16,16,16,16,8,8,8,8,8,4,4,4,4,4] ### ResNet18
            elif args.model == 'DLA':
                width_list = [32 for i in range(9)] + [16 for i in range(14)] + [8 for i in range(14)] + [4 for i in range(6)] ### DLA
            elif args.model == 'SimpleConv':
                width_list = [32, 32] ### two conv layers
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
            T_max = 200

        else:
            if args.model == 'ResNet18':
                width_list = [28,28,28,28,28,14,14,14,14,14,7,7,7,7,7,4,4,4,4,4] ### ResNet18
            elif args.model == 'DLA':
                width_list = [28 for i in range(9)] + [14 for i in range(14)] + [7 for i in range(14)] + [4 for i in range(6)] ### DLA
            elif args.model == 'SimpleConv':
                width_list = [28, 14] ### two conv layers
            optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4, eps=1e-7)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
            T_max = 120

        model_path =  outdir + 'ckpt.pth'
        model_path_test =  outdir + 'ckpt_best_test.pth'

        tr_loss_list = []
        tr_acc_list = []
        ts_loss_list = []
        ts_acc_list = []
        best_keeping_list = []
        print('epoch: ', start_epoch)

        conv_outputs = width_list
        if new_sedghi or gouk_correct:

            if args.dataset == 'cifar':
                if args.model in ['ResNet18', 'DLA']:
                    conv_outputs = [[elm, elm] for elm in width_list]
                else:
                    conv_outputs = get_conv_output_shapes(net, (3,32,32))
            else:
                if args.model in ['ResNet18', 'DLA']:
                    conv_outputs = [[elm, elm] for elm in width_list]
                else:
                    conv_outputs = get_conv_output_shapes(net, (1,28,28))

            print('conv outputs: ', conv_outputs)


        sv_df = {}
        for epoch in range(start_epoch, T_max):
            tr_loss, tr_acc = train(epoch, optimizer, scheduler, criterion, writer=writer, model_path=model_path, width_list=conv_outputs, new_sedghi=new_sedghi, lip4conv=lip4conv, gouk_correct=gouk_correct)
            ts_loss, ts_acc = test(epoch, criterion, writer=writer, model_path=model_path_test)

            if ts_acc == best_acc:
                best_keeping_list.append(1)
            else:
                best_keeping_list.append(0)

            if manual_SV:
                net.zero_grad()
                idx = 0
                for (m_name, m) in net.named_modules():
                    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                        if isinstance(m, torch.nn.Linear):
                            x0 = torch.randn(20, m.weight.shape[1], device=device)
                            const = 'Linear_' + str(idx)
                        elif isinstance(m, torch.nn.Conv2d):
                            # if idx > 0:
                            if idx > 1:
                                continue
                            if args.dataset == 'cifar':
                                if idx == 0:
                                    VT_shape = [1, 3, 32, 32]
                                if idx == 1:
                                    VT_shape = [1, 64, 32, 32]
                            else:
                                VT_shape = [1, 1, 28, 28]
                                if idx == 1:
                                    VT_shape = [1, 64, 28, 28]
                            x0 = torch.randn(VT_shape, device=device)
                            const = 'Conv2d_' + str(idx)

                        qr = power_qr(lambda x: m(x) - m(torch.zeros_like(x)), x0.clone().detach(), n_iters=1000, device=device)

                        if const == 'Linear':
                            print(qr[-1][:20])

                        writer.add_scalar('train/lsv_' + const, qr[-1][0], epoch)
                        if 'lsv_' + const not in sv_df:
                            sv_df['lsv_' + const] = [qr[-1][0].item()]
                        else:
                            sv_df['lsv_' + const].append(qr[-1][0].item())

                        print('lsv_' + const, qr[-1][0])
                        idx += 1

            tr_loss_list.append(tr_loss)
            tr_acc_list.append(tr_acc)
            ts_loss_list.append(ts_loss)
            ts_acc_list.append(ts_acc)

            writer.add_scalar('train/tr_acc', tr_acc, epoch)
            writer.add_scalar('train/ts_acc', ts_acc, epoch)
            writer.add_scalar('train/best_acc', best_acc, epoch)
        
        print('Saving Last..')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'lr': scheduler.get_last_lr(),
        }
        torch.save(state, model_path)

        if manual_SV:
            df = pd.DataFrame(sv_df)
            df.to_csv(outdir + 'sv.csv')

        df = pd.DataFrame({'tr_loss': tr_loss_list, 'tr_acc': tr_acc_list, 'ts_loss': ts_loss_list, 'ts_acc': ts_acc_list, 'best_keeping': best_keeping_list})
        df.to_csv(outdir + 'results' + str(start_epoch) + '.csv')

