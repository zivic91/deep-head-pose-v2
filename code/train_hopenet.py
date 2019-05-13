import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import datasets, hopenet
import torch.utils.model_zoo as model_zoo
import utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=0.00001, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--model', dest='model', help='Type of model to use. For now, ResNet18, ResNet50, ResNet101 and MobileNet are available.',
          default='ResNet50', type=str)
    parser.add_argument('--bin_width_degrees', dest='bin_width_degrees', help='Width of bins in degrees.',
          default=3, type=int)
    parser.add_argument('--unfreeze', dest='unfreeze', help='Choose after how many epochs to unfreeze all layers',
          default=10000, type=int)

    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = parse_args()

    # Load or hard-code parameters
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    gpu = args.gpu_id
    model_type = args.model
    bin_width_degrees = args.bin_width_degrees

    number_of_classes = 198 // bin_width_degrees

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    # Init selected architecture
    if model_type == "MobileNetV2":
        model = hopenet.MobileNetV2(num_classes = number_of_classes)
    elif model_type == 'ResNet18':
        model = hopenet.Hopenet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], number_of_classes)
    elif model_type == 'ResNet101':
        model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], number_of_classes)
    else:
        model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], number_of_classes)

    # TODO: make all combinations (model, snapshot) work
    if args.snapshot == 'from_scratch':
        print "Learning from scratch"
    elif args.snapshot == '':
        print "Loading from model_zoo"
        if model_type == 'ResNet18':
            load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
        elif model_type == 'ResNet101':
            load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'))
        elif model_type == 'ResNet50':
            load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
        else:
            print 'Error: not a valid model name'
    else:
        print "Loading from snapshot"
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Scale(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, transformations, bin_width_degrees)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, transformations, bin_width_degrees)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, transformations, bin_width_degrees)
    elif args.dataset == 'AFLW2000_ds':
        pose_dataset = datasets.AFLW2000_ds(args.data_dir, args.filename_list, transformations, bin_width_degrees)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(args.data_dir, args.filename_list, transformations, bin_width_degrees)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, transformations, bin_width_degrees)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(args.data_dir, args.filename_list, transformations, bin_width_degrees)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(args.data_dir, args.filename_list, transformations, bin_width_degrees)
    else:
        print 'Error: not a valid dataset name'
        sys.exit()

    train_size = int(0.9 * len(pose_dataset))
    print "Training set size: " + str(train_size)
    val_size = len(pose_dataset) - train_size
    print "Validation set size: " + str(val_size)

    idx = list(range(len(pose_dataset)))  # indices to all elements
    random.shuffle(idx)  # in-place shuffle the indices to facilitate random splitting
    train_idx = idx[:train_size]
    val_idx = idx[train_size:]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
    # train_dataset, val_dataset = torch.utils.data.random_split(pose_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               sampler = train_sampler,
                                               batch_size=batch_size,
                                               # shuffle=True,
                                               num_workers=2)

    val_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               sampler = val_sampler,
                                               batch_size=1,
                                               # shuffle=True,
                                               num_workers=2)

    # train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            num_workers=2)

    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax().cuda(gpu)
    idx_tensor = [idx for idx in xrange(number_of_classes)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    print 'Ready to train network.'
    for epoch in range(num_epochs):

        print "Learning rate: " + str(lr)

        if epoch >= args.unfreeze:
            print "First layer unfrozen"
            optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': lr},
                                          {'params': get_non_ignored_params(model), 'lr': lr},
                                          {'params': get_fc_params(model), 'lr': lr * 5}],
                                           lr = args.lr)
        else:
            print "First layer frozen"
            optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                          {'params': get_non_ignored_params(model), 'lr': lr},
                                          {'params': get_fc_params(model), 'lr': lr * 5}],
                                           lr = args.lr)

        lr *= 0.95

        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)

            # Binned labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * bin_width_degrees - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * bin_width_degrees - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * bin_width_degrees - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = [torch.ones(1).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                       %(epoch+1, num_epochs, i+1, len(pose_dataset)//batch_size, loss_yaw.data[0], loss_pitch.data[0], loss_roll.data[0]))


        total = 0
        yaw_error = 0
        pitch_error = 0
        roll_error = 0

        for i, (images, labels, cont_labels, name) in enumerate(val_loader):

            images = Variable(images).cuda(gpu)

            label_yaw = cont_labels[:,0].float()
            label_pitch = cont_labels[:,1].float()
            label_roll = cont_labels[:,2].float()

            if (abs(label_yaw) > 99 or abs(label_pitch) > 99 or abs(label_roll) > 99):
               continue

            total += cont_labels.size(0)

            yaw, pitch, roll = model(images)

            # Binned predictions
            _, yaw_bpred = torch.max(yaw.data, 1)
            _, pitch_bpred = torch.max(pitch.data, 1)
            _, roll_bpred = torch.max(roll.data, 1)

            # Continuous predictions
            yaw_predicted = utils.softmax_temperature(yaw.data, 1)
            pitch_predicted = utils.softmax_temperature(pitch.data, 1)
            roll_predicted = utils.softmax_temperature(roll.data, 1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * bin_width_degrees - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * bin_width_degrees - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * bin_width_degrees - 99

            # Mean absolute error
            yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
            pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
            roll_error += torch.sum(torch.abs(roll_predicted - label_roll))

        print('Validation error in degrees of the model on the ' + str(total) +
        ' validation images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f' % (yaw_error / total,
        pitch_error / total, roll_error / total, (yaw_error + pitch_error + roll_error) / 3 / total))


        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print 'Taking snapshot...'
            torch.save(model.state_dict(),
            'output/snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pkl')
