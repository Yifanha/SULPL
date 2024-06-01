import math
import time

import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os ,torch
import torch.nn as nn
import image_utils
import argparse,random
from tensorboardX import SummaryWriter

import pdb
import matplotlib
import copy
#from model.utils import udata, umath

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default='new2_models/new_model',
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default= None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=10, help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--margin_1', type=float, default=0.15, help='Rank regularization margin. Details described in the paper.')
    parser.add_argument('--margin_2', type=float, default=0.2, help='Relabeling margin. Details described in the paper.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="sgd", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=60, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Drop out rate.')
    parser.add_argument('--gama', type=float, default=0.5, help='Drop out rate.')
    return parser.parse_args()

args = parse_args()
cls2id = {'Surprise': 0,
          'Fear': 1,
          'Disgust': 2,
          'Happiness': 3,
          'Sadness': 4,
          'Anger': 5,
          'Neutral': 6}
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None, basic_aug = False):#这里改了
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.file_paths = []
        self.label = list()
        if self.phase == 'train':
            txt_path = os.path.join(self.raf_path, 'RAFzip//trainvalid//new_train_label_dic.txt')
        else:
            txt_path = os.path.join(self.raf_path, 'RAFzip//trainvalid//new_test_label_dic.txt')
        with open(txt_path, 'r', encoding='utf-8') as rh:
            lines = rh.readlines()
            for line in lines:
                line = line.rstrip()

                path, lab = line.split(' ')
                self.file_paths.append(os.path.join('datasets/raf-basic/RAFzip', path))
                self.label.append(int(lab))
        # self.file_paths = self.file_paths[:1000]
        # self.label = self.label[0:1000]

        self.label = np.array(self.label, dtype=np.int64)

        '''
        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        
        df = pd.read_csv(os.path.join(self.raf_path, 'RAFzip/trainvalid/new_test_label_dic.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)
        '''
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]

    def __len__(self):

        return len(self.file_paths)


    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)

        ori_image = cv2.resize(image, (100, 100))
        ori_image = np.transpose(ori_image, (2, 0, 1))
        image = image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0,1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return ori_image, image, label, idx

class Res18Feature(nn.Module):
    def __init__(self, pretrained = True, num_classes = 7, drop_rate = 0):#改了 为0
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        #####
        resnet = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1
        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
        #####

        #####
        # resnet = models.vgg11(pretrained)
        # self.vgg_features = resnet.features
        # self.vgg_avgpool = resnet.avgpool
        # self.vgg_classifier = resnet.classifier[:-1]
        # self.features = nn.Sequential(self.vgg_features, self.vgg_avgpool)
        # fc_in_dim = self.vgg_classifier[3].out_features
        #####

        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())

    def forward(self, x):
        x = self.features(x) # 64x512x1x1
        if self.drop_rate > 0:
            x =  nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1) # 64x512
        #####
        # x = self.vgg_classifier(x)
        #####
        attention_weights = self.alpha(x)
        out = attention_weights * self.fc(x)
        return attention_weights, out
        
def initialize_weight_goog(m, n=''):
    # weight init as per Tensorflow Official impl
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    # if isinstance(m, CondConv2d):
        # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # init_weight_fn = get_condconv_initializer(
            # lambda w: w.data.normal_(0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        # init_weight_fn(m.weight)
        # if m.bias is not None:
            # m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()
        
def run_training(gama=args.gama):
    writer = SummaryWriter('runs/SCN-TSM without pretrain')
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    import json
    with open(os.path.join(args.checkpoint, 'cls2id.json'), 'w') as fh:
        json.dump(cls2id, fh)
    imagenet_pretrained = True#预训练

    res18 = Res18Feature(pretrained = imagenet_pretrained, drop_rate = args.drop_rate) 
    if not imagenet_pretrained:
         for m in res18.modules():
            initialize_weight_goog(m)

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained) 
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        #model_state_dict = res18.state_dict()
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass
            else:    
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys+=1
                if key in model_state_dict:
                    loaded_keys+=1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict = False)  #这里改动了
        
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # 3x224x224
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))])

    
    train_dataset = RafDataSet(args.raf_path, phase = 'train', transform = data_transforms, basic_aug = True)    
    
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])                                           
    val_dataset = RafDataSet(args.raf_path, phase = 'test', transform = data_transforms_val)    
    print('Validation set size:', val_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    params = res18.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,weight_decay = 1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay = 1e-4)#原为1e-4
    else:
        raise ValueError("Optimizer not supported.")
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    res18 = res18.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    margin_1 = args.margin_1
    margin_2 = args.margin_2
    beta = args.beta
    min_acc_val = 0.6
    global_step = 0
    # writer.add_graph(res18, input_to_model=torch.ones(1, 3, 224, 224).cuda())
    for i in range(1, args.epochs + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        res18.train()
        for batch_i, (ori_imgs, imgs, targets, indexes) in enumerate(train_loader):
            global_step += 1
            batch_sz = imgs.size(0) 
            iter_cnt += 1
            tops = int(batch_sz* beta)
            optimizer.zero_grad()
            imgs = imgs.cuda()

            attention_weights, outputs = res18(imgs)
            #Rank Regularization
            _, top_idx = torch.topk(attention_weights.squeeze(), tops)

            _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest = False)

            high_group = attention_weights[top_idx]
            low_group = attention_weights[down_idx]
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
            diff  = margin_1 - (high_mean - low_mean)
            #diff  = low_mean - high_mean + margin_1

            if diff > 0:
                RR_loss = diff
            else:
                RR_loss = 0.0
            # imgs_01 = imgs.cpu().numpy()
            # imgs_01[:, 0, :, :] = imgs_01[:, 0, :, :] * 0.229 + 0.485
            # imgs_01[:, 1, :, :] = imgs_01[:, 1, :, :] * 0.224 + 0.456
            # imgs_01[:, 2, :, :] = imgs_01[:, 2, :, :] * 0.225 + 0.406
            targets = targets.cuda()
            loss = gama*criterion(outputs, targets) + (1-gama)*RR_loss
            #loss = criterion(outputs, targets)
            _, predicts = torch.max(outputs, 1)
            writer.add_scalar('loss', loss, global_step=global_step)
            # writer.add_histogram('att_weights', attention_weights, global_step=global_step)
            # writer.add_image('input_images', imgs_01[0,:,:,:], global_step=global_step)

            # writer.add_image('input_images', imgs[0,:,:,:], global_step=global_step)
            #writer.add_embedding(outputs, predicts.cpu().numpy().tolist(), ori_imgs, global_step=global_step, tag='img')
            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()

            correct_sum += correct_num

           # Relabel samples
            if i >= args.relabel_epoch:
                sm = torch.softmax(outputs, dim = 1)
                Pmax, predicted_labels = torch.max(sm, 1) # predictions
                Pgt = torch.gather(sm, 1, targets.view(-1,1)).squeeze() # retrieve predicted probabilities of targets
                true_or_false = Pmax - Pgt > margin_2
                update_idx = true_or_false.nonzero().squeeze() # get samples' index in this mini-batch where (Pmax - Pgt > margin_2)
                label_idx = indexes[update_idx] # get samples' index in train_loader
                relabels = predicted_labels[update_idx] # predictions where (Pmax - Pgt > margin_2)
                train_loader.dataset.label[label_idx.cpu().numpy()] = relabels.cpu().numpy() # relabel samples in train_loader
                
        scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt

        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            res18.eval()
            for batch_i, (_, imgs, targets, no_use) in enumerate(val_loader):
                _, outputs = res18(imgs.float().cuda())
                targets = targets.cuda()
                loss = criterion(outputs, targets)
                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_num = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)

            running_loss = running_loss/iter_cnt   
            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(), 4)

            writer.add_scalar('acc', acc, global_step=global_step)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, acc, running_loss))

           
            #if acc > 0.838:

            if acc > min_acc_val:
                min_acc_val = acc
                torch.save({'iter': i,
                            'model_state_dict': res18.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join(args.checkpoint, "epoch"+str(i)+"_acc" + str(min_acc_val) + ".pth"))
                print('Model saved.')



            # torch.save({'iter': i,
            #             'model_state_dict': res18.state_dict(),
             #              'optimizer_state_dict': optimizer.state_dict(),},
            #             os.path.join(args.checkpoint, "epoch"+str(i)+"_acc"+str(acc)+".pth"))
            #print('Model saved.')
# def plot(his_loss, his_acc, his_val_loss, his_val_acc, branch_idx, base_path_his):
#     accuracies_plot = []
#     legends_plot_acc = []
#     losses_plot = [[range(len(his_loss)), his_loss]]
#     legends_plot_loss = ["Training"]
#
#     # Acc
#     for b_plot in range(len(his_acc)):
#         accuracies_plot.append([range(len(his_acc[b_plot])), his_acc[b_plot]])
#         legends_plot_acc.append("Training ({})".format(b_plot + 1))
#
#         accuracies_plot.append([range(len(his_val_acc[b_plot])), his_val_acc[b_plot]])
#         legends_plot_acc.append("Validation ({})".format(b_plot + 1))
#
#     # Ensemble acc
#     accuracies_plot.append([range(len(his_val_acc[-1])), his_val_acc[-1]])
#     legends_plot_acc.append("Validation (E)")
#
#     # Loss
#     for b_plot in range(len(his_val_loss)):
#         losses_plot.append([range(len(his_val_loss[b_plot])), his_val_loss[b_plot]])
#         legends_plot_loss.append("Validation ({})".format(b_plot + 1))
#
#     # Loss
#     umath.plot(losses_plot,
#                title="Training and Validation Losses vs. Epochs for Branch {}".format(branch_idx),
#                legends=legends_plot_loss,
#                file_path=base_path_his,
#                file_name="Loss_Branch_{}".format(branch_idx),
#                axis_x="Training Epoch",
#                axis_y="Loss")
#
#     # Accuracy
#     umath.plot(accuracies_plot,
#                title="Training and Validation Accuracies vs. Epochs for Branch {}".format(branch_idx),
#                legends=legends_plot_acc,
#                file_path=base_path_his,
#                file_name="Acc_Branch_{}".format(branch_idx),
#                axis_x="Training Epoch",
#                axis_y="Accuracy",
#                limits_axis_y=(0.0, 1.0, 0.025))
#
#     # Save plots
#     np.save(path.join(base_path_his, "Loss_Branch_{}".format(branch_idx)), np.array(his_loss))
#     np.save(path.join(base_path_his, "Acc_Branch_{}".format(branch_idx)), np.array(his_acc))
#     np.save(path.join(base_path_his, "Loss_Val_Branch_{}".format(branch_idx)), np.array(his_val_loss))
#     np.save(path.join(base_path_his, "Acc_Val_Branch_{}".format(branch_idx)), np.array(his_val_acc))
if __name__ == "__main__":
    run_training()
