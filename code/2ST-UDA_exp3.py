import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import logging
import random
import sys
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter
from models import *
import utils
from utils import *
from amazon_subset import Subset
import torch.nn.functional as F
from data_prep.fdu_mtl_dataset import get_fdu_mtl_datasets, FduMtlDataset
from vocab import Vocab

# ---------------------- some settings ---------------------- #
random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)


def train(vocab, train_sets, dev_sets, test_sets, threshold, log):

    # 不加my_collate就报错，搞清楚是为什么，报的错是获取数据的错误
    # ---------------------- dataloader ---------------------- #
    my_collate = utils.sorted_collate if opt.model == 'lstm' else utils.unsorted_collate
    # fakelabel_data_collate = utils.fakelabel_data_sorted_collate if opt.model == 'lstm' \
    #     else utils.fakelabel_data_unsorted_collate
    train_loaders = DataLoader(train_sets, opt.batch_size, shuffle=False, collate_fn=my_collate, drop_last=True)
    dev_loaders = DataLoader(dev_sets, opt.batch_size, shuffle=False, collate_fn=my_collate, drop_last=True)
    test_loaders = DataLoader(test_sets, opt.batch_size, shuffle=False, collate_fn=my_collate, drop_last=True)

    # ---------------------- model initialization ---------------------- #
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

    # 转移到gpu上
    F_s, C, D = F_s.to(opt.device), C.to(opt.device), D.to(opt.device)
    for f_d in F_d.values():
        f_d = f_d.to(opt.device)

    # ---------------------- load pre-training model ---------------------- #
    log.info(f'Loading model from {opt.exp3_model_save_file}...')
    F_s.load_state_dict(torch.load(os.path.join(opt.exp3_model_save_file,
                                                f'netF_s.pth')))
    for domain in opt.all_domains:
        if domain in F_d:
            F_d[domain].load_state_dict(torch.load(os.path.join(opt.exp3_model_save_file,
                                                                f'net_F_d_{domain}.pth')))
    C.load_state_dict(torch.load(os.path.join(opt.exp3_model_save_file,
                                              f'netC.pth')))
    D.load_state_dict(torch.load(os.path.join(opt.exp3_model_save_file,
                                              f'netD.pth')))

    log.info('Get fake label:')
    sample_index_list, pseudo_labels, targets_total, normalized_w_total = \
        integrated_genarate_labels(train_loaders, F_s, F_d, C, D, 0.5, log)
    pseudo_labels = pseudo_labels.to(opt.device)
    # 测试一下label预测的准确率
    label_correct_acc = calc_label_prediction(sample_index_list, pseudo_labels, targets_total)
    log.info(f'the correct rate of training data\'s label prediction: {label_correct_acc}')

    # ------------------------------- F_d_target模型以及一些必要的参数定义 ------------------------------- #
    F_d_target = CNNFeatureExtractor(vocab, opt.F_layers, opt.domain_hidden_size,
                                     opt.kernel_num, opt.kernel_sizes, opt.dropout)
    F_d_target = F_d_target.to(opt.device)

    # ---------------------- get fake label(hard label) ---------------------- #
    is_first = True
    co_training_targets, co_training_pseudo_labels = None, None
    init_samples = len(train_sets)
    x = train_sets.X[:]
    y = train_sets.Y[:]
    gen_label_dataset = FduMtlDataset(x, y, opt.max_seq_len)
    print("**************************************")
    print(gen_label_dataset[0])
    print(train_sets[0])
    print("**************************************")
    best_dev_acc = 0.0

    import copy
    total_train_sets_x = []
    total_train_sets_y = None
    odd_step_labels, even_step_labels = init_samples, init_samples
    # 用于记录前后两次的结果，然后再取平均，如果相邻两次产生label的平均数量小于5，则停止
    step = 1

    while threshold > 0.5 and odd_step_labels + even_step_labels > 10:

        # ---------------------- dataloader ---------------------- #
        train_loaders_copy = DataLoader(gen_label_dataset, opt.batch_size, shuffle=False, collate_fn=my_collate, drop_last=True)

        print(len(gen_label_dataset))

        if len(gen_label_dataset) < opt.batch_size:
            break

        # 集成式产生label的方式
        integrated_sample_index_list, integrated_pseudo_labels, targets_total, normalized_w_total = \
            integrated_genarate_labels(train_loaders_copy, F_s, F_d, C, D, threshold, log)
        integrated_pseudo_labels = integrated_pseudo_labels.to(opt.device)

        if integrated_pseudo_labels is None or integrated_pseudo_labels.numel() == 0 or \
                integrated_pseudo_labels.numel() == 1:  # 一个样本都没有的情况或者只有一个样本的情况
            threshold -= 0.02
            continue

        # ------------------------------- 测试一下只用集成方式label预测的准确率 ------------------------------- #
        integrated_label_correct_acc = calc_label_prediction(integrated_sample_index_list,
                                                             integrated_pseudo_labels, targets_total)
        log.info(f'The accuracy of the label produced by the integration method: {integrated_label_correct_acc}')

        # 第一次预测伪label的情况，不需要通过Co-Training，直接产生伪label，然后训练F_d_target即可
        if is_first:
            is_first = False
            final_sample_index_list_index = torch.arange(integrated_sample_index_list.shape[0])
            final_pseudo_labels = integrated_pseudo_labels

        # 非第一次产生label，需要通过Co-Training的方式(第一次产生的label的sample再"扔进"F_d_target中再次预测label，如果与第一次的一致，
        # 那么采用这些label和其对应的sample，否则重新扔到第一次产生label的集合中，回炉重造)
        else:
            integrated_samples_dataset = Subset(gen_label_dataset, integrated_sample_index_list,
                                                integrated_pseudo_labels)
            integrated_samples_dataloader = DataLoader(integrated_samples_dataset, opt.batch_size,
                                                       shuffle=False, collate_fn=my_collate)
            final_sample_index_list_index, final_pseudo_labels = \
                F_d_target_genarate_labels(integrated_samples_dataloader, F_s, F_d_target, C, threshold)

        if final_pseudo_labels is None or final_pseudo_labels.numel() < opt.batch_size:               # 一个样本都没有的情况或者只有一个样本的情况
            break

        if step % 2 == 1:               # 奇数的情况
            odd_step_labels = final_pseudo_labels.numel()
        else:
            even_step_labels = final_pseudo_labels.numel()
        step += 1

        # ------------------------------- 测试一下经过F_d_target过滤后，label准确率为多少 ------------------------------- #
        final_label_correct_acc = calc_label_prediction(integrated_sample_index_list[final_sample_index_list_index],
                                                        final_pseudo_labels, targets_total)
        log.info(f'The accuracy of the label produced by Co-Training: {final_label_correct_acc}')

        if co_training_targets is None:
            co_training_targets = targets_total[integrated_sample_index_list[final_sample_index_list_index]]
            co_training_pseudo_labels = final_pseudo_labels
        else:
            concat_targets = targets_total[integrated_sample_index_list[final_sample_index_list_index]]
            if concat_targets.numel() == 1:
                co_training_targets = torch.cat(
                    [co_training_targets, concat_targets.reshape(1)], 0)
            else:
                co_training_targets = torch.cat(
                    [co_training_targets, concat_targets], 0)
            if final_pseudo_labels.numel() == 1:
                co_training_pseudo_labels = torch.cat(
                    [co_training_pseudo_labels, final_pseudo_labels.reshape(1)], 0)
            else:
                co_training_pseudo_labels = torch.cat(
                    [co_training_pseudo_labels, final_pseudo_labels], 0)

        # ------------------------------- 构造"最终的"target数据集 ------------------------------- #
        # target_dataset = Subset(gen_label_dataset, integrated_sample_index_list[final_sample_index_list_index], final_pseudo_labels)
        # temp_train_sets = gen_label_dataset[integrated_sample_index_list[final_sample_index_list_index]]
        # temp_train_sets = train_sets[integrated_sample_index_list[final_sample_index_list_index]]
        # 数据集不太一样 看一下这里要怎么处理
        # temp_train_sets = gen_label_dataset[integrated_sample_index_list[final_sample_index_list_index]]
        print(integrated_sample_index_list[final_sample_index_list_index])
        x = []
        for i in integrated_sample_index_list[final_sample_index_list_index].numpy().tolist():
            x.append(gen_label_dataset.X[i])

        if y.numel() != 1:
            y = final_pseudo_labels.long().to(opt.device)
        temp_train_sets = FduMtlDataset(x, y, opt.max_seq_len)

        total_train_sets_x.extend(x)
        if total_train_sets_y is None:
            total_train_sets_y = y
        else:
            if y.numel() == 1:
                total_train_sets_y = torch.cat([total_train_sets_y, y.reshape(1)], 0)
            else:
                total_train_sets_y = torch.cat([total_train_sets_y, y], 0)

        # 为什么这里shuffle改成False就不行了？仔细想想看
        # shuffle=False结果与shuffle=True相比结果相差很大，到底是为什么，要想清楚，到底是为什么？我觉得一定是有原因的！
        target_dataloader_labelled = DataLoader(temp_train_sets, opt.batch_size, shuffle=False, collate_fn=my_collate)

        # ------------------------------- 利用最终的target数据集训练F_d_target ------------------------------- #
        best_dev_acc = train_F_d_target(target_dataloader_labelled, dev_loaders, F_s, F_d_target, C, best_dev_acc, log)
        F_d_target.load_state_dict(
            torch.load(os.path.join(opt.exp3_model_save_file, f'net_F_d_target.pth')))

        # ------------------------------- 从gen_label_dataset中删除这些已经有label的数据 ------------------------------- #
        # ------------------------------- 并且组成新的数据集 ------------------------------- #
        all_samples_index = torch.arange(init_samples).numpy()
        # print(all_samples_index.shape)
        # print(final_sample_index_list.shape)
        remain_samples_index = torch.from_numpy(
            np.delete(all_samples_index, integrated_sample_index_list[final_sample_index_list_index]))
        # remain_samples_index = torch.from_numpy(
        #     np.delete(all_samples_index, integrated_sample_index_list))
        print(remain_samples_index.shape[0])
        init_samples = remain_samples_index.shape[0]
        # 暂时试一下这种方式，不行的话再试试其他方式，gen_label_dataset
        # gen_label_dataset = gen_label_dataset[remain_samples_index]    # 这里直接这么写是有误的
        print(len(gen_label_dataset))
        x = []
        for i in remain_samples_index.tolist():
            x.append(gen_label_dataset.X[i])
        y = gen_label_dataset.Y[remain_samples_index]


        # print(x.shape)
        # print(y.shape)
        gen_label_dataset = FduMtlDataset(x, y, opt.max_seq_len)

        # ------------------------------- threshold更新 ------------------------------- #
        threshold -= 0.02
        print("threshold: " + str(threshold))

    # ------------------------------- 测试一下这种方式得到的所有label预测的准确率 ------------------------------- #
    # ------------------------------- 方便与不用Co-Training的方式做对比 ------------------------------- #
    """
    torch.Size([1465])
    torch.Size([1465])
    (1235,)
    """
    print(co_training_targets.shape)
    print(co_training_pseudo_labels.shape)
    equal_idx = torch.nonzero(torch.eq(co_training_targets, co_training_pseudo_labels)).squeeze()
    co_training_label_correct_acc = equal_idx.shape[0] / co_training_targets.shape[0]
    log.info(f'The accuracy of the pseudo label obtained for the first time: {co_training_label_correct_acc}')

    # 最后再利用所有得到的label，协同训练
    opt.max_epoch = 50
    total_train_sets = FduMtlDataset(total_train_sets_x, total_train_sets_y, opt.max_seq_len)
    target_dataloader_labelled = DataLoader(total_train_sets, opt.batch_size, shuffle=True, collate_fn=my_collate)
    # 这里测试一下，分成两个版本，一个版本是更新C，另一个版本是不更新C
    train_F_d_target(target_dataloader_labelled, dev_loaders, F_s, F_d_target, C, best_dev_acc, log)
    # 在测试集上测试
    # 方便对比与单独训练有label时的情况
    log.info('Evaluating test sets:')
    F_d_target.load_state_dict(torch.load(os.path.join(opt.exp3_model_save_file, f'net_F_d_target.pth')))
    # test_acc = evaluate(opt.target_domains, test_loaders, F_s, F_d_target, C)
    # log.info(f'test accuracy: {test_acc}')
    test_acc = evaluate(opt.target_domains, test_loaders, F_s, F_d_target, C, log)
    log.info(f'test accuracy: {test_acc}')

    return test_acc


# 训练F_d_target
def train_F_d_target(target_dataloader_labelled, dev_loaders, F_s, F_d_target, C, best_dev_acc, log):
    optimizer_F_d_target = optim.Adam(F_d_target.parameters(), lr=opt.learning_rate * 2)
    target_train_iters = iter(target_dataloader_labelled)
    # F_s.eval()
    # 为什么C.eval()和C.train()相差这么大？
    # 甚至出现了下述情况，找清楚原因
    # 如果写上C.train()，最后就会出现如下结果，C.eval()或者啥都不写就不会
    # >>> Generate pseudo labels 0, target samples 1
    # C.eval()
    # C.train()
    utils.freeze_net(F_s)
    utils.unfreeze_net(C)
    for epoch in range(opt.max_epoch):
        # training accuracy
        correct, total = 0, 0
        num_iter = len(target_dataloader_labelled)
        for _ in tqdm(range(num_iter)):        # 这里就不该用tdqm，用法错了
            F_d_target.zero_grad()
            inputs, targets = utils.endless_get_next_batch(target_dataloader_labelled, target_train_iters)
            # print("__________________________________________________")
            # print(targets)
            # print("__________________________________________________")
            targets = targets.to(opt.device)
            shared_feat = F_s(inputs)
            domain_feat = F_d_target(inputs)
            features = torch.cat([shared_feat, domain_feat], dim=1)
            c_outputs = C(features)
            l_c = functional.nll_loss(c_outputs, targets)
            # print(l_c)
            l_c.backward()
            _, pred_idx = torch.max(c_outputs, 1)
            # print("**************************************************")
            # print(pred_idx)
            # print("**************************************************")
            total += targets.size(0)
            correct += (pred_idx == targets).sum().item()

            optimizer_F_d_target.step()

        # 暂时留下一个问题，这里没有写验证集，因为如果这里要设置验证集的话，需要将训练集切割
        # 因为unlabeled domain里面只有2000个label sample以及raw_unlabeled_sets(四千多张)
        # 后者训练时用，前者测试的时候用
        # end of epoch
        print("correct is %d" % correct)
        print("total is %d" % total)
        log.info('Ending epoch {}'.format(epoch + 1))
        log.info('Training accuracy:')              # 第一个epoch的accuracy就100%是怎么回事？
        log.info(opt.target_domains)
        log.info(str(100.0 * correct / total))

        # 训练过程中的验证集
        dev_acc = evaluate(opt.target_domains, dev_loaders, F_s, F_d_target, C, log)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            log.info(f'New best dev accuracy: {best_dev_acc}')
            # 保存模型
            torch.save(F_d_target.state_dict(),
                       '{}/net_F_d_target.pth'.format(opt.exp3_model_save_file))

    return best_dev_acc

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
                reshape(inputs[1].shape[0], opt.num_labels * len(opt.source_domains))
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


def F_d_target_genarate_labels(dataloader, F_s, F_d_target, C, threshold):
    """
    :param dataloader: DataLoader
    :param F_s: shared feature extractor
    :param F_d_target: target feature extractor
    :param threshold: Label threshold
    :return:
    """
    F_s.eval()
    C.eval()
    F_d_target.eval()
    it = iter(dataloader)
    pred_values_total, pred_idx_total, targets_total = None, None, None

    for inputs, targets in tqdm(it):
        d_features = F_d_target(inputs)
        shared_features = F_s(inputs)
        c_outputs = torch.exp(C(torch.cat([shared_features, d_features], dim=1)))
        pred_values, pred_idx = torch.max(c_outputs, 1)

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

    # print(pred_values_total.shape)
    # print(targets_total.shape)
    final_sample_index_list, final_pseudo_labels = \
        guess_pseudo_labels(pred_values_total, pred_idx_total, threshold)

    # print(final_sample_index_list.shape)
    # print(final_pseudo_labels.shape)
    # print(final_sample_index_list)
    # print(final_pseudo_labels)
    # print(targets_total[final_sample_index_list])

    # ---------------------- 取交集，得到最后的sample及其相应的label ---------------------- #
    final_pseudo_labels = final_pseudo_labels.to(opt.device)
    equal_idx = torch.nonzero(torch.eq(targets_total[final_sample_index_list], final_pseudo_labels)).squeeze()
    if equal_idx.numel() == 0:
        return None, None
    else:
        # print(equal_idx)
        final_sample_index_list = final_sample_index_list[equal_idx]
        final_pseudo_labels = final_pseudo_labels[equal_idx]
        print(">>> Generate pseudo labels {}, target samples {}".format(
            final_pseudo_labels.numel(), targets_total.shape[0]))
        return final_sample_index_list, final_pseudo_labels


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


def evaluate(name, loader, F_s, F_d_target, C, log):
    F_s.eval()
    if F_d_target:
        F_d_target.eval()
    C.eval()
    it = iter(loader)
    correct = 0
    total = 0
    confusion = ConfusionMeter(opt.num_labels)
    for inputs, targets in tqdm(it):
        targets = targets.to(opt.device)
        if F_d_target:
            d_features = F_d_target(inputs)
        else:
            d_features = torch.zeros(len(targets), opt.domain_hidden_size).to(opt.device)
        features = torch.cat((F_s(inputs), d_features), dim=1)
        outputs = C(features)
        _, pred = torch.max(outputs, 1)
        confusion.add(pred.data, targets.data)
        total += targets.size(0)
        correct += (pred == targets).sum().item()
    acc = correct / total
    log.info('{}: Accuracy on {} samples: {}%'.format(name, total, 100.0*acc))
    log.debug(confusion.conf)
    return acc


def calc_label_prediction(test_sample_index_list, test_pseudo_labels, targets_total):

    test_sets_labels = targets_total[test_sample_index_list]
    equal_idx = torch.nonzero(torch.eq(test_pseudo_labels, test_sets_labels)).squeeze()
    # print(equal_idx.shape[0])
    # print(test_pseudo_labels.shape[0])    有bug
    return equal_idx.numel() / test_pseudo_labels.numel()


def main():
    import copy
    vocab = Vocab(opt.emb_filename)

    all_target_domains = ['MR', 'apparel', 'baby', 'books', 'camera_photo', 'dvd', 'electronics', 'health_personal_care',
                          'imdb', 'kitchen_housewares', 'magazines', 'music', 'software', 'sports_outdoors', 'toys_games', 'video']

    # ---------------------- 一些局部变量的设置 ---------------------- #
    test_acc_dict = {}
    ave_test_acc = 0.0
    threshold = 0.98

    # ---------------------- 设置log ---------------------- #
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
    log = logging.getLogger(__name__)

    for target_domain in all_target_domains:
        # ---------------------- 一些参数的设置 ---------------------- #
        opt.max_epoch = 15
        opt.source_domains = copy.deepcopy(opt.all_domains)
        opt.source_domains.remove(target_domain)
        opt.target_domains = target_domain.split()
        opt.num_labels = FduMtlDataset.num_labels
        base_save_dir = "/hdd/liujian/AAAI-2020_source_code/"
        opt.exp3_model_save_file = base_save_dir + "2020_10_22/source_target_only_unlabeled_data_domain_hidden_size_128"
        opt.exp3_model_save_file = opt.exp3_model_save_file + "/target_" + target_domain

        # ---------------------- 设置log ---------------------- #
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
        log = logging.getLogger(__name__)
        fh = logging.FileHandler(os.path.join(opt.exp3_model_save_file, '2ST-UDA_exp3_2020_11_13.txt'))
        log.addHandler(fh)
        # output options
        log.info(opt)

        # ---------------------- 打印一些必要信息 ---------------------- #
        log.info(opt.domain_hidden_size)
        ### 保证不出错
        if "domain_hidden_size_128" in opt.exp3_model_save_file:
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
        train_sets, dev_sets, test_sets, _, = get_fdu_mtl_datasets(vocab, opt.fdu_mtl_dir, target_domain, opt.max_seq_len)

        # ---------------------- 训练产生伪label和F_d_target的过程 ---------------------- #
        test_acc = train(vocab, train_sets, dev_sets, test_sets, threshold, log)
        test_acc_dict[target_domain] = test_acc

    log.info(f'Training done...')
    log.info(f'test_acc\'s result is: ')
    log.info(test_acc_dict)
    log.info(all_target_domains)
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

