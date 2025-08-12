import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm

import archs2 as archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
import time
import dice_loss
# from medpy.metric.binary import dc
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='DRIVE_UNext_woDS',
                        help='model name')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    return args


def main():
    precision_score = 0
    accuracy_score = 0
    f1_score = 0
    recall_score = 0
    MIoU = 0
    Dice = 0
    k = 0
    TotalTime = 0

    args = parse_args()

    device = torch.device('cuda', args.gpu_id)

    with open('./models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.to(device)

    # Data loading code
    val_path = os.path.join(config['dataset'], 'test', 'images', '*' + config['img_ext'])
    val_img_ids = glob(val_path)
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    val_path2 = os.path.join(config['dataset'], 'test', 'images2', '*' + config['img_ext2'])
    val_img_ids2 = glob(val_path2)
    val_img_ids2 = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids2]


    model.load_state_dict(torch.load('./models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ], additional_targets={
        'image2': 'image'  # 为 mask 添加增强操作
    })

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_ids2=val_img_ids2,
        img_dir=os.path.join(config['dataset'],'test', 'images'),
        img_dir2=os.path.join(config['dataset'], 'test', 'images2'),
        mask_dir=os.path.join(config['dataset'],'test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)


    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, input2, target, meta in tqdm(val_loader, total=len(val_loader)):
            test_target = target
            input = input.to(device)
            input2 = input2.to(device)
            target = target.to(device)
            model = model.to(device)
            # compute output

            if count <= 8:
                start = time.time()
                if config['deep_supervision']:
                    output = model(input, input2)[-1]
                else:
                    output = model(input, input2)
                stop = time.time()

                gput.update(stop - start, input.size(0))

                start = time.time()
                output, _, _ = model(input, input2)  # 步骤1 tensor数据类型

                stop = time.time()

                cput.update(stop - start, input.size(0))
                count = count + 1
                TotalTime = TotalTime + (stop - start)  # 新加的
                total = sum([param.nelement() for param in model.parameters()])  # 新加的

            iou, dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))


            output = torch.sigmoid(output).cpu().numpy()  # 步骤2
            output = np.where(output > 0.5, 1, 0)

            # test_pred = torch.tensor(output)  # 新加的

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

            # 新加的
            # y_pred = torch.round(test_pred)  # 预测
            # y_target = torch.round(test_target)  # 真实目标
            # p = y_pred.cpu().detach().numpy()
            # t = y_target.cpu().detach().numpy()
            # k = k + 1
            # precision_score = precision_score + dice_loss.get_precision(y_pred, y_target, average='binary')
            # accuracy_score = accuracy_score + dice_loss.get_accuracy(y_pred, y_target)
            # f1_score = f1_score + dice_loss.get_f1_score(y_pred, y_target, average='binary')
            # recall_score = recall_score + dice_loss.get_sensitivity(y_pred, y_target, average='binary')
            # MIoU = MIoU + dice_loss.get_MIoU(y_pred, y_target)
            # Dice = Dice + dc(p, t)

        # print("precision_score:")
        # print(precision_score / k)
        # print('\n')
        # print("accuracy_score:")
        # print(accuracy_score / k)
        # print('\n')
        # print("f1_score:")
        # print(f1_score / k)
        # print('\n')
        # print("recall_score:")
        # print(recall_score / k)
        # print('\n')
        # print("k:")
        # print(k)
        # print('\n')
        # print("MIoU:")
        # print(MIoU[1] / k)
        # print('\n')
        # print("Number of parameter: %.2fM" % (total / 1e6))
        # print('\n')
        # print("TestTime:")
        # print(TotalTime)
        # print('\n')
        # # print("Dice:")
        # # print(Dice / k)
    return output
    # 下面这段代码无法到达
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    print('CPU: %.4f' % cput.avg)
    print('GPU: %.4f' % gput.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
