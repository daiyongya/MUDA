import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import os
import random
import sys
from tqdm import tqdm

import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_prep.msda_preprocessed_amazon_dataset import get_msda_amazon_datasets
from models import *

# ---------------------- some settings ---------------------- #
random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)

# 这边代码是只有前向推断的步骤
def train(train_sets, test_sets, log):

    # ---------------------- dataloader ---------------------- #
    train_loaders = DataLoader(train_sets, opt.batch_size, shuffle=False, drop_last=True)
    test_loaders = DataLoader(test_sets, opt.batch_size, shuffle=False, drop_last=True)

    # ---------------------- models的初始化 ---------------------- #
    F_s = None
    F_d = {}
    C, D = None, None
    if opt.model.lower() == 'mlp':
        F_s = MlpFeatureExtractor(opt.feature_num, opt.F_hidden_sizes,
                                  opt.shared_hidden_size, opt.dropout, opt.F_bn)
        for domain in opt.source_domains:
            F_d[domain] = MlpFeatureExtractor(opt.feature_num, opt.F_hidden_sizes,
                                              opt.domain_hidden_size, opt.dropout, opt.F_bn)
    else:
        raise Exception(f'Unknown model architecture {opt.model}')

    C = SentimentClassifier(opt.C_layers, opt.shared_hidden_size + opt.domain_hidden_size,
                            opt.shared_hidden_size + opt.domain_hidden_size, opt.num_labels,
                            opt.dropout, opt.C_bn)
    D = DomainClassifier(opt.D_layers, opt.shared_hidden_size, opt.shared_hidden_size,
                         len(opt.all_domains), opt.loss, opt.dropout, opt.D_bn)

    # 转移到gpu上
    F_s, C, D = F_s.to(opt.device), C.to(opt.device), D.to(opt.device)
    for f_d in F_d.values():
        f_d = f_d.to(opt.device)

    # ---------------------- load pre-training model ---------------------- #
    log.info(f'Loading model from {opt.exp2_model_save_file}...')
    F_s.load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                                f'netF_s.pth')))
    for domain in opt.all_domains:
        if domain in F_d:
            F_d[domain].load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                                                f'net_F_d_{domain}.pth')))
    C.load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                              f'netC.pth')))
    D.load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                              f'netD.pth')))

    log.info('Get fake label for training data:')
    sample_index_list, pseudo_labels, targets_total, normalized_w_total = \
        integrated_genarate_labels(train_loaders, F_s, F_d, C, D, 0.5, log)
    pseudo_labels = pseudo_labels.to(opt.device)
    # 测试一下label预测的准确率
    training_data_label_correct_acc = calc_label_prediction(sample_index_list, pseudo_labels, targets_total)
    log.info(f'the correct rate of training data\'s label prediction: {training_data_label_correct_acc}')

    log.info('Get fake label for test data:')
    sample_index_list, pseudo_labels, targets_total, normalized_w_total = \
        integrated_genarate_labels(test_loaders, F_s, F_d, C, D, 0.5, log)
    pseudo_labels = pseudo_labels.to(opt.device)
    # 测试一下label预测的准确率
    test_data_label_correct_acc = calc_label_prediction(sample_index_list, pseudo_labels, targets_total)
    log.info(f'the correct rate of test data\'s label prediction: {test_data_label_correct_acc}')

    return training_data_label_correct_acc, test_data_label_correct_acc


"""
"集成式"label产生方式，通过c*w来筛选
"""
def integrated_genarate_labels(dataloader, F_s, F_d, C, D, threshold, log):
    """
    :param dataloader: DataLoader
    :param F_s: shared feature extractor
    :param F_d: Private feature extractor
    :param C: Sentiment classifier
    :param D: Domain classifier
    :param threshold: Label threshold   从0.98 ==> 0.5
    :return:
    """
    """Genrate pseudo labels for unlabeled domain dataset."""
    # ---------------------- Switch to eval() mode ---------------------- #
    print(threshold)
    F_s.eval()
    C.eval()
    for f_d in F_d.values():
        f_d.eval()
    D.eval()

    it = iter(dataloader)               # batch_size为32
    # print(len(it))

    F_d_features = []
    F_d_outputs = []
    normalized_w_total = None  # 用于保存target domain所有句子对应的w(经过softmax之后的)
    pred_values_total, pred_idx_total, targets_total = None, None, None

    with torch.no_grad():
        for inputs, targets in tqdm(it):

            # ---------------------- target samples经过F_d得到的F_d_features ---------------------- #
            # print(len(F_d))
            for f_d in F_d.values():
                F_d_features.append(f_d(inputs))

            # ---------------------- F_d_features经过D之后的"分数" ---------------------- #
            private_feature_d_outputs = []
            for f_d_feature in F_d_features:
                private_feature_d_outputs.append(torch.exp(D(f_d_feature)).cpu())       # 得到15个16维的向量(取一个做分析)

            # print(private_feature_d_outputs)
            # ---------------------- 去掉target domain这一维度 ---------------------- #
            # ---------------------- 剩余部分再取softmax ---------------------- #   怎么样分清楚谁跟我像？并拉开不跟我像的差距？margin？归一化？
            # 剩余source 总：0.7 某个分量: 0.4    总：0.1 某个分量: 0.07   ===> 1.全部归一化 2.top几(可以top3)归一化？ 其余0.00001之类的微小值 (先尝试)
            tgt_idx = None
            for i in range(len(opt.all_domains)):
                if opt.all_domains[i] == opt.target_domains[0]:
                    tgt_idx = i
                    break

            # print("target index is: ")
            # print(tgt_idx)
            # print(opt.all_domains)
            # # 输出target domain的分数在各个F_d的分数
            # for feature_d_output, domain in zip(private_feature_d_outputs, F_d.keys()):
            #     print(domain)
            #     max_index = np.argmax(feature_d_output.cpu().numpy(), 1)
            #     print(max_index)
            #     print(opt.all_domains[max_index[0]])
            #     for i in range(feature_d_output.shape[0]):
            #         print(feature_d_output[i, max_index[i]])
            #     print("___________________________________")

            # 15 * 15  ==>   1.全部归一化比较合适，这样不会平滑
            # ---------------------- 将去掉target domain这一维度的softmax值进行归一化 ---------------------- #
            new_private_feature_d_outputs = []
            for feature_d_output in private_feature_d_outputs:
                feature_d_output = feature_d_output[:, torch.arange(feature_d_output.shape[1]) != tgt_idx]
                # normalized_feature_d_output = F.softmax(feature_d_output, 1)
                normalized_feature_d_output = feature_d_output / torch.cat([torch.sum(feature_d_output, 1, keepdim=True)
                                                                            for _ in range(len(opt.source_domains))], 1)   # 利用归一化的方式
                new_private_feature_d_outputs.append(normalized_feature_d_output)

            # ---------------------- 得到w，取自于每个F_d_features经过D之后的"分数" ---------------------- #
            # ---------------------- 每个"分数"取该domain相应于 ---------------------- #
            # print(new_private_feature_d_outputs[0].shape)
            w = []
            for i in range(len(opt.source_domains)):
                w.append(new_private_feature_d_outputs[i][:, i])

            # ---------------------- 将w经过softmax归一化 ---------------------- #
            w = torch.stack(w, 1)
            # print(w.shape)
            normalized_w = F.softmax(w, 1)
            # print(normalized_w.shape)
            # print(np.argmax(normalized_w.cpu().numpy(), 1))

            if normalized_w_total is None:
                normalized_w_total = normalized_w
            else:
                normalized_w_total = torch.cat([normalized_w_total, normalized_w], 0)

            # ---------------------- concat(F_d_features, shared_feature)经过C之后的分数 ---------------------- #
            c_outputs = []
            shared_feature = F_s(inputs)
            for f_d_feature in F_d_features:
                features = torch.cat((shared_feature, f_d_feature), dim=1)
                c_outputs.append(torch.exp(C(features)).cpu())

            # ---------------------- 得到c * w后的分数 ---------------------- #
            normalized_w = torch.from_numpy(np.repeat(normalized_w.cpu().numpy(), repeats=opt.num_labels, axis=1))
            # print(normalized_w.shape)
            F_d_outputs_tensor = torch.stack(c_outputs, 1).\
                reshape(inputs.shape[0], opt.num_labels * len(opt.source_domains))
            # print(F_d_outputs_tensor)
            # print(F_d_outputs_tensor.shape)
            c_mul_w = normalized_w * F_d_outputs_tensor

            # ---------------------- 对c_mul_w处理得到hard label ---------------------- #
            even_indices = torch.LongTensor(np.arange(0, 2 * len(opt.source_domains), 2))
            odd_indices = torch.LongTensor(np.arange(1, 2 * len(opt.source_domains), 2))
            even_index_scores = torch.index_select(c_mul_w, 1, even_indices)
            odd_index_scores = torch.index_select(c_mul_w, 1, odd_indices)
            # print(even_index_scores.shape)
            # print(odd_index_scores.shape)
            even_index_scores_sum = torch.sum(even_index_scores, 1).unsqueeze(1)
            odd_index_scores_sum = torch.sum(odd_index_scores, 1).unsqueeze(1)
            # print(even_index_scores_sum)
            # print(odd_index_scores_sum)
            pred_scores = torch.cat([even_index_scores_sum, odd_index_scores_sum], 1)
            pred_values, pred_idx = torch.max(pred_scores, 1)

            # ---------------------- 保存结果 ---------------------- #
            if pred_values_total is None:
                pred_values_total = pred_values
                pred_idx_total = pred_idx
                targets_total = targets
            else:
                pred_values_total = torch.cat(
                    [pred_values_total, pred_values], 0)
                pred_idx_total = torch.cat(
                    [pred_idx_total, pred_idx], 0)
                targets_total = torch.cat(
                    [targets_total, targets], 0)

            F_d_features.clear()
            F_d_outputs.clear()

    # print(pred_values_total.shape)
    # print(targets_total.shape)
    sample_index_list, pseudo_labels = \
        guess_pseudo_labels(pred_values_total, pred_idx_total, threshold)

    log.info(">>> Generate pseudo labels {}, target samples {}".format(
        pseudo_labels.numel(), targets_total.shape[0]))

    return sample_index_list, pseudo_labels, targets_total, normalized_w_total


def guess_pseudo_labels(pred_values, pred_idx_total, threshold):
    filtered_idx = torch.arange(0, pred_values.shape[0])[pred_values.gt(threshold)]
    # test_filtered_idx = torch.arange(0, pred_values.shape[0])[pred_values.gt(0.95)]
    print("**************************************************")
    # print(pred_values)
    # print(test_filtered_idx.shape)     # 第一次：torch.Size([338])，第二次：torch.Size([771])
    print(filtered_idx.shape)
    print("**************************************************")
    pred_idx_total = pred_idx_total.cpu()
    pseudo_labels = torch.index_select(pred_idx_total, 0, filtered_idx)
    return filtered_idx, pseudo_labels


def calc_label_prediction(test_sample_index_list, test_pseudo_labels, targets_total):

    test_sets_labels = targets_total[test_sample_index_list]
    equal_idx = torch.nonzero(torch.eq(test_pseudo_labels, test_sets_labels)).squeeze()
    # print(equal_idx.shape[0])
    # print(test_pseudo_labels.shape[0])    有bug
    return equal_idx.numel() / test_pseudo_labels.numel()


def main():
    import copy

    all_target_domains = ['books', 'dvd', 'electronics', 'kitchen']
    opt.all_domains = copy.deepcopy(all_target_domains)

    # ---------------------- 设置log ---------------------- #
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
    log = logging.getLogger(__name__)

    test_acc_dict, train_acc_dict = {}, {}


    for target_domain in all_target_domains:
        # ---------------------- 一些参数的设置 ---------------------- #
        opt.source_domains = copy.deepcopy(opt.all_domains)

        print(opt.all_domains)
        print(opt.source_domains)

        opt.source_domains.remove(target_domain)
        opt.target_domains = target_domain.split()
        opt.num_labels = 2
        opt.model = 'mlp'
        opt.dataset = 'prep-amazon'
        base_save_dir = "/hdd/liujian/AAAI-2020_source_code/"
        opt.exp2_model_save_file = base_save_dir + "2020_11_16/source_target_only_unlabeled_data_domain_hidden_size_128_exp2"
        opt.exp2_model_save_file = opt.exp2_model_save_file + "/target_" + target_domain

        # ---------------------- 设置log ---------------------- #
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
        log = logging.getLogger(__name__)
        fh = logging.FileHandler(os.path.join(opt.exp2_model_save_file, 'WS-UDA_log.txt'))
        log.addHandler(fh)
        # output options
        log.info(opt)


        # ---------------------- 打印一些必要信息 ---------------------- #
        log.info(opt.domain_hidden_size)
        ### 保证不出错
        if "domain_hidden_size_128" in opt.exp2_model_save_file:
            assert opt.shared_hidden_size == opt.domain_hidden_size
        ### 保证source和target不出错
        assert set(opt.source_domains + opt.target_domains) == set(opt.all_domains)
        ### 对于有无bn的情况下都要保证不出错
        assert opt.C_bn is True
        assert opt.D_bn is True
        log.info(opt.all_domains)
        log.info(opt.source_domains)
        log.info(opt.target_domains)

        # ---------------------- 加载数据集 ---------------------- #
        log.info(f'Loading {opt.dataset} Datasets...')
        datasets, raw_unlabeled_sets = get_msda_amazon_datasets(
                opt.prep_amazon_file, target_domain, opt.feature_num)
        log.info(f'Done Loading {opt.dataset} Datasets.')
        log.info(f'Domains: {opt.all_domains}')


        train_sets, test_sets = datasets, raw_unlabeled_sets

        # ---------------------- 训练产生伪label和F_d_target的过程 ---------------------- #
        training_data_label_correct_acc, test_data_label_correct_acc = train(train_sets, test_sets, log)
        train_acc_dict[target_domain] = training_data_label_correct_acc
        test_acc_dict[target_domain] = test_data_label_correct_acc

    ave_train_acc, ave_test_acc = 0.0, 0.0
    log.info(f'Training done...')

    log.info(f'train_acc\'s result is: ')
    log.info(train_acc_dict)
    log.info(all_target_domains)
    for key in train_acc_dict:
        log.info(str(key) + ": " + str(train_acc_dict[key]))
        ave_train_acc += train_acc_dict[key]

    log.info(ave_train_acc)
    ave_train_acc = ave_train_acc / len(all_target_domains)
    log.info(f'ave_train_acc\'s result is: ')
    log.info(ave_train_acc)

    log.info(f'test_acc\'s result is: ')
    log.info(test_acc_dict)
    log.info(all_target_domains)
    for key in test_acc_dict:
        log.info(str(key) + ": " + str(test_acc_dict[key]))
        ave_test_acc += test_acc_dict[key]

    log.info(ave_test_acc)
    ave_test_acc = ave_test_acc / len(all_target_domains)
    log.info(f'ave_test_acc\'s result is: ')
    log.info(ave_test_acc)


if __name__ == '__main__':
    main()

