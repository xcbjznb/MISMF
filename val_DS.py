import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import arch_itc3 as arch_itc
from dataset import Dataset
from metrics import iou_score
import dice_loss
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from arch_itc import UNext


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='DRIVE_UNext_woDS',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    args.name = 'DRIVE_UNext_woDS'
    with open('./models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = arch_itc.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

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
    'image2': 'image'    
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
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()




    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, input2, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            input2 = input2.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output, _, _,_ = model(input, input2)


            iou,dice = iou_score(output, target)


            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))



            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0





            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)


    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
