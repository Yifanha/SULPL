import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os, torch
import torch.nn as nn
import image_utils
import argparse, random
import pdb
import random
from PIL import Image
import scipy.misc
import matplotlib
from sklearn.model_selection import KFold


def read_im(path):
    """
    Read image in RGB mode
    :param path: image src path
    :return: RGB image
    """
    im = cv2.imread(path, 1)
    if im is None:
        im = scipy.misc.imread(path, mode='RGB')
    else:
        return im[:, :, ::-1]
    return im

##1是no pretrain 2是no relabel
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--raf_path', type=str, default='D:\\Facial Expression Recognition\\表情识别数据集\\CK+',
    #                     help='Raf-DB dataset path.')
    parser.add_argument('--raf_path', type=str,
                        default='D:\\Facial Expression Recognition\\表情识别数据集\\CK+',
                        help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default='models/Oulu-CASIA no pretrain',
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=10,
                        help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--margin_1', type=float, default=0.15,
                        help='Rank regularization margin. Details described in the paper.')
    parser.add_argument('--margin_2', type=float, default=0.2,
                        help='Relabeling margin. Details described in the paper.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=6, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=7, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    return parser.parse_args()


args = parse_args()


def get_file_list(p, exts=None):
    """
    Find image files in test data path
    :return: list of absolute path of the images
    """
    files = []
    for parent, dirnames, filenames in os.walk(p):
        for filename in filenames:
            if exts is not None:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
            else:
                files.append(os.path.join(parent, filename))

    return files


def get_images(p):
    """
    Find image files in test data path
    :return: list of absolute path of the images
    """
    return get_file_list(p, exts=['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'gif', 'bmp'])


# cls2id = {'disgust': 0,
#           'happy': 1,
#           'normal': 2,
#           'surprised': 3}
class RafDataSet(data.Dataset):
    def __init__(self, data_list, class_names, phase, transform=None, basic_aug=False):

        self.phase = phase
        self.transform = transform
        self.file_paths = data_list
        self.label = list()
        sorted(class_names)
        cls2id = {c: i for i, c in enumerate(class_names)}

        import json
        with open(os.path.join(args.checkpoint, 'cls2id.json'), 'w') as fh:
            json.dump(cls2id, fh)

        self.label = [cls2id[os.path.basename(os.path.dirname(fp))] for fp in self.file_paths]

        self.label = np.array(self.label, dtype=np.int64)
        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):

        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # image = cv2.imread(path, 1)
        # image = image[:, :, ::-1]  # BGR to RGB
        image = np.array(Image.open(path))
        if len(image.shape) == 2:
            image = np.dstack((image, image, image))
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


class Res18Feature(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)  # 64x512x1x1

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)  # 64x512

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


def run_training():
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    imagenet_pretrained = False

    res18 = Res18Feature(pretrained=imagenet_pretrained, drop_rate=args.drop_rate)
    if not imagenet_pretrained:
        for m in res18.modules():
            initialize_weight_goog(m)

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained)
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
                pass
            else:
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys += 1
                if key in model_state_dict:
                    loaded_keys += 1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict=False)

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 3x224x224
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))])

    params = res18.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported.")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    res18 = res18.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    margin_1 = args.margin_1
    margin_2 = args.margin_2
    beta = args.beta
    k = 10
    kf = KFold(k, shuffle=True, random_state=1234)
    image_list = get_images(args.raf_path)
    class_names = os.listdir(os.path.join(args.raf_path))
    accs = []
    for k_idx, (train_index, test_index) in enumerate(kf.split(image_list)):
        train_list = [image_list[i] for i in train_index]
        test_list = [image_list[i] for i in test_index]
        train_dataset = RafDataSet(train_list, class_names, phase='train', transform=data_transforms, basic_aug=True)

        print('Train set size:', train_dataset.__len__())
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True)

        data_transforms_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        val_dataset = RafDataSet(test_list, class_names, phase='test', transform=data_transforms_val)
        print('Validation set size:', val_dataset.__len__())

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.workers,
                                                 shuffle=False,
                                                 pin_memory=True)
        best_val_acc = 0.0
        for i in range(1, args.epochs + 1):
            running_loss = 0.0
            correct_sum = 0
            iter_cnt = 0
            res18.train()
            for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
                batch_sz = imgs.size(0)
                iter_cnt += 1
                tops = int(batch_sz * beta)
                optimizer.zero_grad()
                imgs = imgs.cuda()

                attention_weights, outputs = res18(imgs)
                # Rank Regularization
                _, top_idx = torch.topk(attention_weights.squeeze(), tops)

                _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest=False)

                high_group = attention_weights[top_idx]
                low_group = attention_weights[down_idx]
                high_mean = torch.mean(high_group)
                low_mean = torch.mean(low_group)
                # diff  = margin_1 - (high_mean - low_mean)
                diff = low_mean - high_mean + margin_1

                if diff > 0:
                    RR_loss = diff
                else:
                    RR_loss = 0.0

                targets = targets.cuda()
                loss = criterion(outputs, targets) + RR_loss
                loss.backward()
                optimizer.step()

                running_loss += loss
                _, predicts = torch.max(outputs, 1)
                correct_num = torch.eq(predicts, targets).sum()
                correct_sum += correct_num

                # Relabel samples
                if i >= args.relabel_epoch:
                    sm = torch.softmax(outputs, dim=1)
                    Pmax, predicted_labels = torch.max(sm, 1)  # predictions
                    Pgt = torch.gather(sm, 1,
                                       targets.view(-1, 1)).squeeze()  # retrieve predicted probabilities of targets
                    true_or_false = Pmax - Pgt > margin_2
                    update_idx = true_or_false.nonzero().squeeze()  # get samples' index in this mini-batch where (Pmax - Pgt > margin_2)
                    label_idx = indexes[update_idx]  # get samples' index in train_loader
                    relabels = predicted_labels[update_idx]  # predictions where (Pmax - Pgt > margin_2)
                    train_loader.dataset.label[
                        label_idx.cpu().numpy()] = relabels.cpu().numpy()  # relabel samples in train_loader

            scheduler.step()
            acc = correct_sum.float() / float(train_dataset.__len__())
            running_loss = running_loss / iter_cnt
            print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))

            with torch.no_grad():
                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0
                res18.eval()
                for batch_i, (imgs, targets, _) in enumerate(val_loader):
                    _, outputs = res18(imgs.cuda())
                    targets = targets.cuda()
                    loss = criterion(outputs, targets)
                    running_loss += loss
                    iter_cnt += 1
                    _, predicts = torch.max(outputs, 1)
                    correct_num = torch.eq(predicts, targets)
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += outputs.size(0)

                running_loss = running_loss / iter_cnt
                acc = bingo_cnt.float() / float(sample_cnt)
                acc = np.around(acc.numpy(), 4)
                print("[Fold %d] [Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (k_idx, i, acc, running_loss))

                if acc > best_val_acc:
                    # save_path = 'models/fer'
                    # if not os.path.exists(save_path):
                    #     os.makedirs(save_path)
                    torch.save({'model_state_dict': res18.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(), },
                               os.path.join(args.checkpoint, "K-" + str(k_idx) + ".pth"))
                    print('Model saved.')
                    best_val_acc = acc
        accs.append(best_val_acc)
    print('k-folder accs:', accs)
    print('k-folder mean acc:', np.array(accs).mean())


if __name__ == "__main__":
    run_training()
