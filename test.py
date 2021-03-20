import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import ImagerLoader  # our data_loader
import numpy as np
from trijoint import *
import pickle
from args import get_parser
import os
import random
# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.manual_seed(opts.seed)

np.random.seed(opts.seed)

if not (torch.cuda.device_count()):
    device = torch.device(*('cpu', 0))
else:
    torch.cuda.manual_seed(opts.seed)
    device = torch.device(*('cuda', 0))

def rank_emb(result_path, type_embedding, test_size):
    random.seed(1234)
    with open(result_path + 'img_embeds.pkl', 'rb') as f:
        im_vecs = pickle.load(f)
    with open(result_path + 'rec_embeds.pkl', 'rb') as f:
        instr_vecs = pickle.load(f)
    with open(result_path + 'rec_ids.pkl', 'rb') as f:
        names = pickle.load(f)
    print(type_embedding)
    # Sort based on names to always pick same samples for medr
    idxs = np.argsort(names)
    names = names[idxs]

    # Ranker
    N = test_size
    idxs = range(N)

    glob_rank = []
    results = ''
    glob_med = []
    glob_avg = []
    glob_std = []


    glob_recall = {1:0.0,5:0.0,10:0.0}
    for i in range(10):
        results += 'the {}th test \n'.format(i)
        ids = random.sample(range(0,len(names)), N)
        im_sub = im_vecs[ids,:]
        instr_sub = instr_vecs[ids,:]
        ids_sub = names[ids]
        rank_list = []
        med_rank = []
        recall = {1:0.0,5:0.0,10:0.0}

        for ii in idxs:
            distance = {}
            for j in range(N):
                if type_embedding == 'i2r':
                    distance[j] = np.linalg.norm(im_sub[ii] - instr_sub[j])  # for im2recipe
                else:
                    distance[j] = np.linalg.norm(instr_sub[ii] - im_sub[j])  # for recipe2im
            distance_sorted = sorted(distance.items(), key=lambda x:x[1])
            pos = np.where(np.array(distance_sorted) == distance[ii])[0][0]

            if (pos + 1) == 1:
                recall[1] += 1
            if (pos + 1) <= 5:
                recall[5] += 1
            if (pos + 1) <= 10:
                recall[10] += 1

            # store the position
            rank_list.append(pos + 1)

        for i in recall.keys():
            recall[i]=(recall[i] + 0.0)/N

        med = np.median(rank_list)
        # print("median", med)
        results += 'median: {} \t'.format(med)

        std = np.std(rank_list)
        # print('std', std)
        results += 'std: {} \t'.format(std)

        avg = np.average(rank_list)
        # print("average", avg)
        results += 'average: {} \n'.format(avg)

        results += 'recall: {} \n'.format(recall)

        for i in recall.keys():
            glob_recall[i] += recall[i]
        glob_med.append(med)
        glob_avg.append(avg)
        glob_std.append(std)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i] / 10
    final_med = np.average(glob_med)
    final_std = np.average(glob_std)
    final_avg = np.average(glob_avg)
    print("Mean median", final_med)
    print("Mean standard deviation", final_std)
    print("Mean average", final_avg)

    print("Recall", glob_recall)

    final_result = "median: {} \n".format(final_med) + "std: {} \n".format(final_std) + "Average: {} \n".format(final_avg) \
                   + "Recall: {} \n".format(glob_recall) + results
    with open(result_path + type_embedding + '{}.txt'.format(N), 'w') as f:
        f.write(final_result)

def do_test():
    result_dir = opts.result_dir
   
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    model.to(device) 

    model_path = opts.snapshots + '{}.pth.tar'.format(result_dir)
    print("=> loading checkpoint '{}'".format(model_path))
    if device.type == 'cpu':
        checkpoint = torch.load(model_path, encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(model_path, encoding='latin1')
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, checkpoint['epoch']))

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # preparing test loader
    test_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
                     transforms.Compose([
                         transforms.Scale(256),  # rescale the image keeping the original aspect ratio
                         transforms.CenterCrop(224),  # we get only the center of that rescaled
                         transforms.ToTensor(),
                         normalize,
                     ]), data_path=opts.data_path, sem_reg=opts.semantic_reg, partition='test'),
        batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.workers, pin_memory=True)
    print('Test loader prepared.')

    # run test
    test(test_loader, model, result_dir)
    
    result_path = 'results/{}/'.format(result_dir)
    print('1k test')
    rank_emb(result_path, type_embedding='i2r', test_size=1000)
    rank_emb(result_path, type_embedding='r2i', test_size=1000)

    print('10k test')
    rank_emb(result_path, type_embedding='i2r', test_size=10000)
    rank_emb(result_path, type_embedding='r2i', test_size=10000)


def test(test_loader, model, result_dir):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        with torch.no_grad():
            input_var = list()
            for j in range(len(input)):
                input_var.append(input[j].to(device))
            target_var = list()
            for j in range(len(target) - 2):  # we do not consider the last two objects of the list
                target_var.append(target[j].to(device))

            # compute output
            output = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4])
            img_id_fea = output[0]
            rec_id_fea = output[1]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i == 0:
                data0 = img_id_fea.data.cpu().numpy()
                data1 = rec_id_fea.data.cpu().numpy()
                data2 = target[-2]
                data3 = target[-1]
            else:
                data0 = np.concatenate((data0, img_id_fea.data.cpu().numpy()), axis=0)
                data1 = np.concatenate((data1, rec_id_fea.data.cpu().numpy()), axis=0)
                data2 = np.concatenate((data2, target[-2]), axis=0)
                data3 = np.concatenate((data3, target[-1]), axis=0)

    if not os.path.exists(opts.path_results + result_dir):
        os.mkdir(opts.path_results + result_dir)
    print(result_dir)
    with open(opts.path_results + result_dir + '/img_embeds.pkl', 'wb') as f:
        pickle.dump(data0, f)
    with open(opts.path_results + result_dir + '/rec_embeds.pkl', 'wb') as f:
        pickle.dump(data1, f)
    with open(opts.path_results + result_dir + '/img_ids.pkl', 'wb') as f:
        pickle.dump(data2, f)
    with open(opts.path_results + result_dir + '/rec_ids.pkl', 'wb') as f:
        pickle.dump(data3, f)

if __name__ == '__main__':
    result_dir = opts.result_dir
    print(result_dir)
    # do_test()

    result_path = 'results/{}/'.format(result_dir)
    print('1k test')
    rank_emb(result_path, type_embedding='i2r', test_size=1000)
    rank_emb(result_path, type_embedding='r2i', test_size=1000)

    print('10k test')
    rank_emb(result_path, type_embedding='i2r', test_size=10000)
    rank_emb(result_path, type_embedding='r2i', test_size=10000)