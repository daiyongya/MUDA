import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from collections import defaultdict
import itertools
import logging
import os
import pickle
import random
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchnet.meter import ConfusionMeter

from options import opt

# 设置随机种子
random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)

from data_prep.fdu_mtl_dataset import get_fdu_mtl_datasets, FduMtlDataset
from models import *
import utils
from vocab import Vocab

# 这个是第一阶段的训练过程
def train(vocab, train_sets, dev_sets, test_sets, unlabeled_sets, log):
    """
    train_sets, dev_sets, test_sets: dict[domain] -> AmazonDataset
    For unlabeled domains, no train_sets are available
    """
    # dataset loaders
    train_loaders, unlabeled_loaders = {}, {}
    train_iters, unlabeled_iters = {}, {}
    dev_loaders, test_loaders = {}, {}
    my_collate = utils.sorted_collate if opt.model == 'lstm' else utils.unsorted_collate

    # 各个源域有训练集，验证集和测试集
    for domain in opt.source_domains:
        train_loaders[domain] = DataLoader(train_sets[domain],
                                           opt.batch_size, shuffle=True, collate_fn=my_collate)
        train_iters[domain] = iter(train_loaders[domain])

        dev_loaders[domain] = DataLoader(dev_sets[domain],
                                         opt.batch_size, shuffle=False, collate_fn=my_collate)
        test_loaders[domain] = DataLoader(test_sets[domain],
                                          opt.batch_size, shuffle=False, collate_fn=my_collate)

    # 源域和目标域均有无标签的数据集
    for domain in opt.all_domains:
        if domain in opt.target_domains:
            uset = unlabeled_sets[domain]
        else:
            # for labeled domains, consider which data to use as unlabeled set
            if opt.unlabeled_data == 'both':
                uset = ConcatDataset([train_sets[domain], unlabeled_sets[domain]])
            elif opt.unlabeled_data == 'unlabeled':
                uset = unlabeled_sets[domain]
            elif opt.unlabeled_data == 'train':
                uset = train_sets[domain]
            else:
                raise Exception(f'Unknown options for the unlabeled data usage: {opt.unlabeled_data}')
        unlabeled_loaders[domain] = DataLoader(uset,
                                               opt.batch_size, shuffle=True, collate_fn=my_collate)
        unlabeled_iters[domain] = iter(unlabeled_loaders[domain])

    # ---------------------- models的初始化 ---------------------- #
    F_s = None
    F_d = {}
    C, D = None, None
    F_s = CNNFeatureExtractor(vocab, opt.F_layers, opt.shared_hidden_size,
                              opt.kernel_num, opt.kernel_sizes, opt.dropout)
    for domain in opt.source_domains:
        F_d[domain] = CNNFeatureExtractor(vocab, opt.F_layers, opt.domain_hidden_size,
                                          opt.kernel_num, opt.kernel_sizes, opt.dropout)

    C = SentimentClassifier(opt.C_layers, opt.shared_hidden_size + opt.domain_hidden_size,
                            opt.shared_hidden_size + opt.domain_hidden_size, opt.num_labels,
                            opt.dropout, opt.C_bn)
    D = DomainClassifier(opt.D_layers, opt.shared_hidden_size, opt.shared_hidden_size,
                         len(opt.all_domains), opt.loss, opt.dropout, opt.D_bn)

    F_s, C, D = F_s.to(opt.device), C.to(opt.device), D.to(opt.device)
    for f_d in F_d.values():
        f_d = f_d.to(opt.device)
    # optimizers
    optimizer = optim.Adam(itertools.chain(*map(list, [F_s.parameters() if F_s else [], C.parameters()] + [f.parameters() for f in F_d.values()])),
                           lr=opt.learning_rate)
    optimizerD = optim.Adam(D.parameters(), lr=opt.D_learning_rate)

    # training
    best_acc, best_avg_acc = defaultdict(float), 0.0
    for epoch in range(opt.max_epoch):
        F_s.train()
        C.train()
        D.train()
        for f in F_d.values():
            f.train()

        # training accuracy
        correct, total = defaultdict(int), defaultdict(int)
        # D accuracy
        d_correct, d_total = 0, 0
        private_d_correct, private_d_total = 0, 0
        # conceptually view 1 epoch as 1 epoch of the first domain
        num_iter = len(train_loaders[opt.source_domains[0]])
        for i in tqdm(range(num_iter)):
            # D iterations
            utils.freeze_net(F_s)
            map(utils.freeze_net, F_d.values())
            utils.freeze_net(C)
            utils.unfreeze_net(D)
            # WGAN n_critic trick since D trains slower
            n_critic = opt.n_critic
            if opt.wgan_trick:
                if opt.n_critic > 0 and ((epoch == 0 and i < 25) or i % 500 == 0):
                    n_critic = 100

            #### 记得这里的不同之处，D训练的时候还要加上private loss
            for _ in range(n_critic):
                D.zero_grad()
                loss_d = {}
                # train on both labeled and unlabeled domains
                # 这里是在所有领域
                for domain in opt.all_domains:
                    # 数据本身的label没有用到
                    d_inputs, _ = utils.endless_get_next_batch(
                        unlabeled_loaders, unlabeled_iters, domain)

                    # 注意这里是获取domain label
                    d_targets = utils.get_domain_label(opt.loss, domain, len(d_inputs[1]))
                    shared_feat = F_s(d_inputs)

                    # 加上private loss
                    if domain in opt.source_domains:
                        private_feat = F_d[domain](d_inputs)
                        private_d_outputs = D(private_feat)
                        _, private_pred = torch.max(private_d_outputs, 1)
                        private_d_total += len(d_inputs[1])

                    d_outputs = D(shared_feat)
                    # D accuracy
                    _, pred = torch.max(d_outputs, 1)
                    d_total += len(d_inputs[1])

                    if opt.loss.lower() == 'l2':
                        _, tgt_indices = torch.max(d_targets, 1)
                        d_correct += (pred == tgt_indices).sum().item()
                        shared_l_d = functional.mse_loss(d_outputs, d_targets)
                        private_l_d = 0.0
                        if domain in opt.source_domains:
                            private_d_correct += (private_pred == tgt_indices).sum().item()
                            private_l_d = functional.mse_loss(private_d_outputs, d_targets)

                        l_d = shared_l_d + private_l_d
                        l_d.backward()

                    else:
                        d_correct += (pred == d_targets).sum().item()
                        shared_l_d = functional.nll_loss(d_outputs, d_targets)
                        private_l_d = 0.0
                        if domain in opt.source_domains:
                            private_d_correct += (private_pred == d_targets).sum().item()
                            private_l_d = functional.nll_loss(private_d_outputs, d_targets)

                        l_d = shared_l_d + private_l_d
                        l_d.backward()

                    loss_d[domain] = l_d.item()
                optimizerD.step()

            # F&C iteration
            utils.unfreeze_net(F_s)
            map(utils.unfreeze_net, F_d.values())
            utils.unfreeze_net(C)
            utils.freeze_net(D)
            if opt.fix_emb:
                utils.freeze_net(F_s.word_emb)
                for f_d in F_d.values():
                    utils.freeze_net(f_d.word_emb)
            F_s.zero_grad()
            for f_d in F_d.values():
                f_d.zero_grad()
            C.zero_grad()

            # 更新C，公有特征提取器和私有特征提取器的参数，只在source domain上更新
            for domain in opt.source_domains:
                inputs, targets = utils.endless_get_next_batch(
                    train_loaders, train_iters, domain)
                targets = targets.to(opt.device)
                shared_feat = F_s(inputs)
                domain_feat = F_d[domain](inputs)
                features = torch.cat((shared_feat, domain_feat), dim=1)
                c_outputs = C(features)
                l_c = functional.nll_loss(c_outputs, targets)
                l_c.backward(retain_graph=True)
                _, pred = torch.max(c_outputs, 1)
                total[domain] += targets.size(0)
                correct[domain] += (pred == targets).sum().item()

            # update F_s with D gradients on all domains
            for domain in opt.all_domains:
                d_inputs, _ = utils.endless_get_next_batch(
                    unlabeled_loaders, unlabeled_iters, domain)
                shared_feat = F_s(d_inputs)
                d_outputs = D(shared_feat)
                if opt.loss.lower() == 'gr':
                    d_targets = utils.get_domain_label(opt.loss, domain, len(d_inputs[1]))
                    l_d = functional.nll_loss(d_outputs, d_targets)
                    if opt.lambd > 0:
                        l_d *= -opt.lambd
                elif opt.loss.lower() == 'bs':
                    d_targets = utils.get_random_domain_label(opt.loss, len(d_inputs[1]))
                    l_d = functional.kl_div(d_outputs, d_targets, size_average=False)
                    if opt.lambd > 0:
                        l_d *= opt.lambd
                elif opt.loss.lower() == 'l2':
                    d_targets = utils.get_random_domain_label(opt.loss, len(d_inputs[1]))
                    l_d = functional.mse_loss(d_outputs, d_targets)
                    if opt.lambd > 0:
                        l_d *= opt.lambd
                l_d.backward()

            optimizer.step()

        # end of epoch
        log.info('Ending epoch {}'.format(epoch + 1))
        if d_total > 0:
            log.info('D Training Accuracy: {}%'.format(100.0 * d_correct / d_total))
        log.info('Training accuracy:')
        log.info('\t'.join(opt.source_domains))
        log.info('\t'.join([str(100.0 * correct[d] / total[d]) for d in opt.source_domains]))

        # 验证集上的准确率
        log.info('Evaluating validation sets:')
        acc = {}
        for domain in opt.source_domains:
            acc[domain] = evaluate(domain, dev_loaders[domain],
                                   F_s, F_d[domain] if domain in F_d else None, C, log)
        avg_acc = sum([acc[d] for d in opt.source_domains]) / len(opt.source_domains)
        log.info(f'Average validation accuracy: {avg_acc}')

        # 测试集上的准确率
        log.info('Evaluating test sets:')
        test_acc = {}
        for domain in opt.source_domains:
            test_acc[domain] = evaluate(domain, test_loaders[domain],
                                        F_s, F_d[domain] if domain in F_d else None, C, log)
        avg_test_acc = sum([test_acc[d] for d in opt.source_domains]) / len(opt.source_domains)
        log.info(f'Average test accuracy: {avg_test_acc}')

        if avg_acc > best_avg_acc:
            log.info(f'New best average validation accuracy: {avg_acc}')
            best_acc['valid'] = acc
            best_acc['test'] = test_acc
            best_avg_acc = avg_acc
            with open(os.path.join(opt.exp3_model_save_file, 'options.pkl'), 'wb') as ouf:
                pickle.dump(opt, ouf)
            torch.save(F_s.state_dict(),
                       '{}/netF_s.pth'.format(opt.exp3_model_save_file))
            for d in opt.source_domains:
                if d in F_d:
                    torch.save(F_d[d].state_dict(),
                               '{}/net_F_d_{}.pth'.format(opt.exp3_model_save_file, d))
            torch.save(C.state_dict(),
                       '{}/netC.pth'.format(opt.exp3_model_save_file))
            torch.save(D.state_dict(),
                       '{}/netD.pth'.format(opt.exp3_model_save_file))

    # end of training
    log.info(f'Best average validation accuracy: {best_avg_acc}')
    return best_acc


def evaluate(name, loader, F_s, F_d, C, log):
    F_s.eval()
    if F_d:
        F_d.eval()
    C.eval()
    it = iter(loader)
    correct = 0
    total = 0
    confusion = ConfusionMeter(opt.num_labels)
    for inputs, targets in tqdm(it):
        targets = targets.to(opt.device)
        if not F_d:
            # target domain
            d_features = torch.zeros(len(targets), opt.domain_hidden_size).to(opt.device)
        else:
            d_features = F_d(inputs)
        features = torch.cat((F_s(inputs), d_features), dim=1)
        outputs = C(features)
        _, pred = torch.max(outputs, 1)
        confusion.add(pred.data, targets.data)
        total += targets.size(0)
        correct += (pred == targets).sum().item()
    acc = correct / total
    log.info('{}: Accuracy on {} samples: {}%'.format(name, total, 100.0 * acc))
    log.debug(confusion.conf)
    return acc


def main():
    import copy
    vocab = Vocab(opt.emb_filename)

    all_target_domains = ['MR', 'apparel', 'baby', 'books', 'camera_photo', 'dvd', 'electronics', 'health_personal_care',
                          'imdb', 'kitchen_housewares', 'magazines', 'music', 'software', 'sports_outdoors', 'toys_games', 'video']
    opt.all_domains = copy.deepcopy(all_target_domains)

    for target_domain in all_target_domains:

        # ---------------------- 一些参数的设置 ---------------------- #
        #### 注意这里source和target都用同一批数据
        opt.unlabeled_data = 'unlabeled'
        opt.source_domains = copy.deepcopy(opt.all_domains)

        print(opt.all_domains)
        print(opt.source_domains)

        opt.source_domains.remove(target_domain)
        opt.target_domains = target_domain.split()
        opt.num_labels = FduMtlDataset.num_labels
        base_save_dir = "/hdd/liujian/AAAI-2020_source_code/"
        opt.exp3_model_save_file = base_save_dir + "2020_10_22/source_target_only_unlabeled_data_domain_hidden_size_128"
        opt.exp3_model_save_file = opt.exp3_model_save_file + "/target_" + target_domain
        if not os.path.exists(opt.exp3_model_save_file):
            os.makedirs(opt.exp3_model_save_file)

        # ---------------------- 设置log ---------------------- #
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
        log = logging.getLogger(__name__)
        fh = logging.FileHandler(os.path.join(opt.exp3_model_save_file, 'log.txt'))
        log.addHandler(fh)
        # output options
        log.info(opt)

        # ---------------------- 加载各个domain的数据集 ---------------------- #
        train_sets, dev_sets, test_sets, unlabeled_sets = {}, {}, {}, {}
        for domain in opt.all_domains:
            train_sets[domain], dev_sets[domain], test_sets[domain], unlabeled_sets[domain] = \
                get_fdu_mtl_datasets(vocab, opt.fdu_mtl_dir, domain, opt.max_seq_len)
        log.info(f'Done Loading {opt.dataset} Datasets.')

        # ---------------------- 打印一些必要信息 ---------------------- #
        log.info(opt.domain_hidden_size)
        ### 保证不出错
        if "domain_hidden_size_128" in opt.exp3_model_save_file:
            assert opt.shared_hidden_size == opt.domain_hidden_size
        ### 保证source和target不出错
        assert set(opt.source_domains + opt.target_domains) == set(opt.all_domains)
        ### 保证不出错
        assert opt.C_bn is True
        assert opt.D_bn is True
        ### 这一块保证不能出错
        assert opt.unlabeled_data == 'unlabeled'

        log.info(opt.unlabeled_data)
        log.info(opt.all_domains)
        log.info(opt.source_domains)
        log.info(opt.target_domains)

        cv = train(vocab, train_sets, dev_sets, test_sets, unlabeled_sets, log)
        log.info(f'Training done...')
        acc = sum(cv['valid'].values()) / len(cv['valid'])
        log.info(f'Validation Set Domain Average\t{acc}')
        test_acc = sum(cv['test'].values()) / len(cv['test'])
        log.info(f'Test Set Domain Average\t{test_acc}')


if __name__ == '__main__':
    main()