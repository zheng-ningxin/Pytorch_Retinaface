from __future__ import print_function
import sys
import datetime
import time
import math
from layers.modules import MultiBoxLoss
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
import torch.utils.data as data
import torch.optim as optim
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from utils.model_parse import mask_decorater
from utils.filter_pruner import filter_pruner
from widerface_evaluate.evaluation import evaluation
import json
import shutil

#import sensitivity.sensitivity_pruner as sensitivity_pruner
# from sensitivity.sensitivity_pruner import SensitivityPruner
from nni.compression.torch.sensitivity_pruner import SensitivityPruner

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50',
                    help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str,
                    help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/',
                    type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true",
                    default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder',
                    default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02,
                    type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4,
                    type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true",
                    default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float,
                    help='visualization_threshold')
parser.add_argument('--analysis_start', default=0, type=int,
                    help='Only analyze the layers start from')
parser.add_argument('--analysis_end', default=None, type=int,
                    help='The sensitivity analysis stops at this layer')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=24, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3,
                    type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--training_dataset',
                    default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--batch_size', default=6, type=int,
                    help='batch_size for finetune')
parser.add_argument('--iter', type=int, default=1,
                    help='maximum iteration of the sensitivity pruner')

parser.add_argument('--sensitivity', default=None,
                    help='resume from the sensitivity results')
parser.add_argument('--checkpoint', default=None,
                    help='resume the previous pruned weights')
parser.add_argument('--pruner_cfg', default=None,
                    help='resume the previous pruner config')

parser.add_argument('--outdir', help='save the result in this directory')
parser.add_argument('--target_ratio', type=float, default=0.5,
                    help='Target ratio of the remained weights compared to the original model')
parser.add_argument('--maxiter', type=int, default=None,
                    help='max iteration of the sentivity pruning')
parser.add_argument('--finetune_epoch', type=int, default=1,
                    help='Finetune Epoch')
parser.add_argument('--ratio_step', type=float, default=0.1,
                    help='the amount of the pruned weight in each prune iteration')
parser.add_argument('--threshold', type=float, default=0.05,
                    help='The accuracy drop threshold during the sensitivity analysis')
parser.add_argument('--lr_decay', type=float,
                    default=0.5, help='lr_decay rate')

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def val(net):
    # remove the last evaluation result
    if os.path.exists('./widerface_evaluate/widerface_txt'):
        shutil.rmtree('./widerface_evaluate/widerface_txt')
    # testing dataset
    net.eval()
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    with torch.no_grad():
        for i, img_name in enumerate(test_dataset):

            image_path = testset_folder + img_name
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = np.float32(img_raw)

            # testing scale
            target_size = 1600
            max_size = 2150
            im_shape = img.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            resize = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)
            if args.origin_size:
                resize = 1

            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize,
                                 fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor(
                [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            _t['forward_pass'].tic()
            loc, conf, landms = net(img)  # forward pass
            _t['forward_pass'].toc()
            _t['misc'].tic()
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0),
                                  prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            # dets = dets[:args.keep_top_k, :]
            # landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)
            _t['misc'].toc()

            # --------------------------------------------------------------------
            save_name = args.save_folder + img_name[:-4] + ".txt"
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, "w") as fd:
                bboxs = dets
                file_name = os.path.basename(save_name)[:-4] + "\n"
                bboxs_num = str(len(bboxs)) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for box in bboxs:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + \
                        " " + str(h) + " " + confidence + " \n"
                    fd.write(line)

            # print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

            # save image
            if args.save_image:
                for b in dets:
                    if b[4] < args.vis_thres:
                        continue
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    cv2.rectangle(img_raw, (b[0], b[1]),
                                  (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] + 12
                    cv2.putText(img_raw, text, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    # landms
                    cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                    cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                    cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
                # save image
                if not os.path.exists("./results/"):
                    os.makedirs("./results/")
                name = "./results/" + str(i) + ".jpg"
                cv2.imwrite(name, img_raw)
    acc = evaluation('./widerface_evaluate/widerface_txt',
                     './widerface_evaluate/ground_truth/')
    print(acc)
    return sum(acc)/3.0


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50
rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
#batch_size = cfg['batch_size']
batch_size = args.batch_size  # we use single gpu to train
max_epoch = args.finetune_epoch
gpu_train = cfg['gpu_train']
num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder
priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(net):
    torch.set_grad_enabled(True)

    optimizer = optim.SGD(net.parameters(), lr=initial_lr,
                          momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, args.lr_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()
    net.train()
    epoch = 0
    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0
    start_iter = 0

    for iteration in range(start_iter, max_iter):
        #print('Iteration:', iteration)
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(
                dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder +
                           cfg['name'] + '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        # lr = adjust_learning_rate(
        #     optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        # print(torch.cuda.memory_stats(device))
        # print(images.size())
        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        # # block the grad of the pruned filter
        # #with torch.no_grad():
        # layer.weight.grad.data.mul_(layer.w_mask)
        # if hasattr(layer, 'bias') and layer.bias is not None:
        #     layer.bias.grad.data.mul_(layer.b_mask)

        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        if iteration % epoch_size == 0:
            print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                  .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                          epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr_scheduler.get_lr()[0], batch_time, str(datetime.timedelta(seconds=eta))))
            lr_scheduler.step()
    print('training end')
    # sys.exit(1)


if __name__ == '__main__':

    # torch.set_grad_enabled(False)
    # net and model
    net = RetinaFace(cfg=cfg)
    net = load_model(net, args.trained_model, args.cpu)

    print('Finished loading model!')
    # print(net)
    #cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    pruner = SensitivityPruner(net, val, train)
    if args.checkpoint and args.pruner_cfg:
        pruner.resume(args.checkpoint, args.pruner_cfg)
    func_args = [net]
    pruner.compress(args.target_ratio, val_args=func_args, finetune_args=func_args, threshold=args.threshold, ratio_step=args.ratio_step,
                    MAX_ITERATION=args.iter, checkpoint_dir=args.outdir, resume_sensitivity=args.sensitivity)

    model_file = '%s_sen_prune_%.2f_step_%.2f_iter_%d.pth' % (
        args.network, args.target_ratio, args.ratio_step, args.iter)
    pruner_cfg = '%s_prune_cfg_%.2f_step_%.2f_iter_%d.json' % (
        args.network, args.target_ratio, args.ratio_step, args.iter)
    model_file = os.path.join(args.outdir, model_file)
    pruner_cfg = os.path.join(args.outdir, pruner_cfg)
    os.makedirs(args.outdir, exist_ok=True)
    pruner.export(model_file, pruner_cfg)
    # also save the sensitivity_result
    sensitivity_file = os.path.join(args.outdir, 'sensitivity.json')
    with open(sensitivity_file, 'w') as jf:
        json.dump(pruner.sensitivities, jf)
