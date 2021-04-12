from tqdm import tqdm
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from utils import *
from torch.utils import data

#from Adaptive_VGG import MainModel
from models.Adaptive_VGG import MainModel

def train_model(model, device, db_gen, optimizer, epoch, file):
    model.train()
    corr1 = 0
    corr5 = 0
    top1 = 0
    top5 = 0

    with tqdm(total=len(db_gen), ncols=120) as pbar:
        for idx_ct, (m_batch, m_label) in enumerate(db_gen):
            m_batch = m_batch.to(device)
            m_label = m_label.to(device)

            outputs = model(m_batch, m_label)  # output
            loss = criterion(outputs, m_label)
            corr1, corr5 = accuracy(outputs, m_label, topk=(1, 5))
            top1 += corr1
            top5 += corr5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx_ct == 0:
                for p in optimizer.param_groups:
                    lr_cur = p['lr']
                    break

            pbar.set_description('Epoch%d : cce:%.3f, cur_lr:%.6f, top1_acc:%.3f, top5_acc:%.3f' % (epoch, loss, float(lr_cur), top1/(idx_ct+1), top5/(idx_ct+1)))
            pbar.update(1)

        if bool(parser['do_lr_decay']):
            if parser['lr_decay'].lower() == 'step':
                lr_scheduler.step()
        
        file.write('Epoch%d : cce:%.3f, cur_lr:%.6f, top1_acc:%.5f, top5_acc:%.5f \n' % (epoch, loss, float(lr_cur), top1/(idx_ct+1), top5/(idx_ct+1)))


def evaluate_model_iden(model, db_gen, device, save_dir, epoch):
    model.eval()
    counter = 0
    tot_loss = 0
    top1 = 0
    top5 = 0
    with torch.set_grad_enabled(False):
        # 1st, extract speaker embeddings.
        with tqdm(total=len(db_gen), ncols=120) as pbar:
            for (m_batch, m_label) in db_gen:
                m_batch = m_batch.to(device)
                m_label = m_label.to(device)
                outputs = model(m_batch, m_label)

                loss = criterion(outputs, m_label)
                tot_loss += loss

                corr1, corr5 = accuracy(outputs, m_label, topk=(1, 5))
                top1 += corr1
                top5 += corr5
                counter += 1
                
                if bool(parser['do_lr_decay']):
                    if parser['lr_decay'].lower() == 'rlr':
                        lr_scheduler.step(tot_loss/counter)
                
                pbar.update(1)
            
            pbar.set_description('(Test) top-1: %.3f, top-5: %.5f' % (top1/counter, top5/counter))
            pbar.update(1)
            

    return top1/counter, top5/counter, tot_loss/counter


def evaluate_model_veri(model, db_gen, device, l_utt, save_dir, epoch, l_trial):
    model.eval()
    with torch.set_grad_enabled(False):
        # 1st, extract speaker embeddings.
        l_embeddings = []
        with tqdm(total=len(db_gen), ncols=120) as pbar:
            for m_batch in db_gen:
                m_batch = m_batch.to(device)
                code = model(m_batch)
                l_embeddings.extend(code.cpu().numpy()) #>>> (batchsize, codeDim)
                pbar.update(1)
        
            d_embeddings = {}
            if not len(l_utt) == len(l_embeddings):
                print(len(l_utt), len(l_embeddings))
                exit()
            for k, v in zip(l_utt, l_embeddings):
                d_embeddings[k] = v

            y_score = [] # score for each sample
            y = [] # label for each sample 
            for line in l_trial:
                trg, utt_a, utt_b = line.strip().split(' ')
                utt_a = utt_a[:-4]
                utt_b = utt_b[:-4]
                y.append(int(trg))
                y_score.append(cos_sim(d_embeddings[utt_a], d_embeddings[utt_b]))
            eer = ComputeEER(y_score, y)

            p_target = 0.01
            c_miss = 1
            c_fa = 1
            mindcf, thres = ComputeMinDcf(y_score, y, p_target, c_miss, c_fa)

            pbar.set_description('(Test) eer: %.3f, MinDCF: %.5f, threshold: %.5f' % (eer, mindcf, thres))
            pbar.update(1)
                
    return eer, mindcf, thres


def get_utts_iden(l_trial_iden):
    l_utt_train = []
    l_utt_test = []

    for line in l_trial_iden:
        set_num, utt = line.strip().split(' ')
        utt = utt[:-4]
        if int(set_num) == 3:
            l_utt_test.append(utt)
        else :
            l_utt_train.append(utt)

    return sorted(l_utt_train), sorted(l_utt_test)


def get_utts_veri(l_trial_veri, src_dir):
    l_utt_train = []
    l_utt_test = []

    for line in l_trial_veri:
        _, utt_a, utt_b = line.strip().split(' ')
        utt_a = utt_a[:-4]
        utt_b = utt_b[:-4]
        if utt_a not in l_utt_test: l_utt_test.append(utt_a)
        if utt_b not in l_utt_test: l_utt_test.append(utt_b)

    for r, _, fs in os.walk(src_dir):
        base = '/'.join(r.split('/')[-2:])+'/'
        for f in fs:
            if f[-3:] != 'npy':
                continue

            if (int(r.split('/')[-2][2:]) < 10270) or (int(r.split('/')[-2][2:]) > 10309):
                l_utt_train.append(base+f[:-4])

    return sorted(l_utt_train), sorted(l_utt_test)


def get_label_dic(l_utt_train):
    d_label = {}
    idx_counter = 0
    for utt in l_utt_train:
        spk = utt.split('/')[0]
        if spk not in d_label:
            d_label[spk] = idx_counter
            idx_counter += 1

    return d_label


class Dataset_VoxCeleb(data.Dataset):
    def __init__(self, list_IDs, base_dir, nb_time=0, labels={}, cut=True, return_label=True):
        '''
        self.list_IDs    : list of strings (each string: utt key)
        self.labels      : dictionary (key: utt key, value: label integer)
        self.nb_time     : integer, the number of timesteps for each mini-batch
        cut              : (boolean) adjust utterance duration for mini-batch construction
        return_label     : (boolean) 
        '''
        self.list_IDs = list_IDs
        self.nb_time = nb_time
        self.base_dir = base_dir
        self.labels = labels
        self.cut = cut
        self.return_label = return_label

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = np.load(self.base_dir+ID+'.npy')
        
        nb_time = X.shape[-1]
        if self.cut:
            if nb_time >= self.nb_time:
                start_idx = np.random.randint(low=0,
                                              high=nb_time - self.nb_time)
                X_cut = X[:, start_idx:start_idx+self.nb_time]
            else:
                raise ValueError('short time input')
        else:
            X_cut = X

        if self.return_label:
            y = self.labels[ID.split('/')[0]]
            return torch.FloatTensor(X_cut), y
        else : 
            return torch.FloatTensor(X_cut)


if __name__ == '__main__':
    ##### load yaml file & set comet_ml config #####
    _abspath = os.path.abspath(__file__)
    dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser = yaml.load(f_yaml, Loader=yaml.FullLoader)

    ##### setting seed #####
    torch.manual_seed(parser['seed'])
    np.random.seed(parser['seed'])

    ##### device setting ######
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current cuda device ', torch.cuda.current_device()) 

    if parser['mode'] == 'veri' :
        print('======== Speaker Verification Task =======')
        # get utt_lists & define labels
        with open('./list/veri_test.txt', 'r') as ff:
            l_trial_veri = ff.readlines()
        l_train, l_test = get_utts_veri(l_trial_veri, parser['DB'])

    elif parser['mode'] == 'iden' :
        print('======= Speaker Identification Task ======')
        
        # get utt_lists & define labels
        with open('./list/iden_split.txt', 'r') as ff:
            l_trial_iden = ff.readlines()
        l_train, l_test = get_utts_iden(l_trial_iden)
        
    else :
        raise NameError('Undefined mode of task')

    d_label = get_label_dic(l_train)

    print('Train num : %d' % (len(l_train)))
    print('Test num : %d' % (len(l_test)))
    print('Train Class num : %d' % (len(d_label)))

    ##### Mdefine model #####
    print('=============== Model Info ===============')
    # Set model
    model, utt_time = MainModel(nOut = parser['nOut'], classlen = len(d_label), mode = parser['mode'], nc_divide = parser['nc_divide'], encoder_type = parser['encoder_type']) #nn.DataParallel
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print('nb_params : {}'.format(nb_params))
    print('==========================================')
    model = nn.DataParallel(model)
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    ##### define dataset #####
    trainset = Dataset_VoxCeleb(list_IDs=l_train,
                                labels=d_label,
                                nb_time=utt_time,
                                base_dir=parser['DB'],
                                cut=True)
    trainset_gen = data.DataLoader(trainset,
                                   batch_size=parser['batch_size'],
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=8)

    if parser['mode'] == 'veri' :
        testset = Dataset_VoxCeleb(list_IDs=l_test,
                                cut=False,
                                return_label = False,
                                labels=d_label,
                                nb_time=utt_time,
                                base_dir=parser['DB'])
    else : 
        testset = Dataset_VoxCeleb(list_IDs=l_test,
                                cut=False,
                                labels=d_label,
                                nb_time=utt_time,
                                base_dir=parser['DB'])
    testset_gen = data.DataLoader(testset,
                                 batch_size=1,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=0)


    ##### save directory #####
    save_dir = parser['save_dir'] + parser['name'] + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + 'models/'):
        os.makedirs(save_dir + 'models/')


    ##### optimizer #####
    if parser['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=parser['lr'],
                                    momentum=parser['opt_mom'],
                                    weight_decay=parser['wd'],
                                    nesterov=bool(parser['nesterov']))
        print('Optimizer type : SGD (lr = %.2f, momentum = %.3f, wd = %.4f, nesterov = %d)'%(parser['lr'], parser['opt_mom'], parser['wd'], parser['amsgrad']))

    elif parser['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=parser['lr'],
                                     weight_decay=parser['wd'],
                                     amsgrad=bool(parser['amsgrad']))
        print('Optimizer type : Adam (lr = %.2f, wd = %.4f, amsgrad = %d)'%(parser['lr'], parser['wd'], parser['amsgrad']))

    else:
        raise NotImplementedError('Add other optimizers if needed')


    ##### learning rate scheduler #####
    if bool(parser['do_lr_decay']):
        if parser['lr_decay'].lower() == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=parser['step_size'],
                                                           gamma=parser['gamma'])

            print('Step learning rate decay (step size : %d, gamma : %.3f)'%(parser['step_size'],parser['gamma']))

        elif parser['lr_decay'].lower() == 'rlr':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                      factor=parser['factor'],
                                                                      patience=parser['patience'],
                                                                      threshold=parser['threshold'],
                                                                      verbose=bool(parser['verbose']))
            print('RLR learning rate decay (factor : %.3f)'%(parser['factor']))

        else:
            raise NotImplementedError('Add other scheduler if needed')

    else :
        print('Without learning rate decay')

    print('==========================================')



    ############## TRAINING ###############
    best_eer = 50
    do_test = 0
    f_ccr = open(save_dir + 'eers.txt', 'a', buffering=1)
    f_best = open(save_dir + 'best.txt', 'a', buffering=1)

    for epoch in (range(parser['epoch'])):
        ##### train phase #####
        train_model(model=model,
                    device=device,
                    db_gen=trainset_gen,
                    optimizer=optimizer,
                    epoch=epoch,
                    file=f_ccr)

        ##### test phase #####
        if (epoch % (parser['do_test']) == 0) or (epoch >= parser['do_test']):
            if parser['mode'] == 'iden':
                top1_acc, top5_acc, ccr_val = evaluate_model_iden(model=model,
                                                                db_gen=testset_gen,
                                                                device=device,
                                                                save_dir=save_dir,
                                                                epoch=epoch)
                f_ccr.write('(Test) ccr: %.5f, Top-1 accuracy: %.5f, Top-5 accuracy: %.5f \n' %(ccr_val, top1_acc, top5_acc))

                save_model_dict = model.module.state_dict()

                if bool(parser['save_best_only']):
                    if top1_acc >= best_eer:
                        print('New best test accuracy: %.5f' % top1_acc)
                        best_eer = top1_acc
                        # save best model
                        torch.save(save_model_dict, save_dir + 'models/best.pt')
                        torch.save(optimizer.state_dict(), save_dir + 'models/best_opt.pt')
                        f_best.write('(Epoch%d) Test_accuray : %.5f\n' %(epoch, top1_acc))
                else :
                    if top1_acc >= best_eer:
                        print('New best test accuracy: %.5f' % top1_acc)
                    torch.save(save_model_dict, save_dir + 'models/model_epoch_%d.pt'%(epoch))
                    torch.save(optimizer.state_dict(), save_dir + 'models/opt_epoch_%d.pt'%(epoch))

            if parser['mode'] == 'veri':
                eer, mindcf, thres = evaluate_model_veri(model = model,
                                                    db_gen = testset_gen, 
                                                    device = device,
                                                    l_utt = l_test,
                                                    save_dir = save_dir,
                                                    epoch = epoch,
                                                    l_trial = l_trial_veri)
                f_ccr.write('(Test) eer: %.5f, MinDCF: %.5f, thres: %.5f \n'%(eer, mindcf, thres))

                save_model_dict = model.module.state_dict()

                if bool(parser['save_best_only']):
                    if eer <= best_eer:
                        print('New best test accuracy: %.5f'%(eer))
                        best_eer = eer
                        #save best model
                        torch.save(save_model_dict, save_dir +  'models/best.pt')
                        torch.save(optimizer.state_dict(), save_dir + 'models/best_opt.pt')
                        f_best.write('(Epoch%d) Test_accuray : %.5f, MinDCF: %.5f, thres: %.5f\n'%(epoch, eer, mindcf, thres))
                else :
                    if top1_acc >= best_eer:
                        print('New best test accuracy: %.5f' % top1_acc)
                    torch.save(save_model_dict, save_dir + 'models/model_epoch_%d.pt'%(epoch))
                    torch.save(optimizer.state_dict(), save_dir + 'models/opt_epoch_%d.pt'%(epoch))

    f_ccr.close()
    f_best.close()
