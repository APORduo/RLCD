import torch
import torch.distributed as dist
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.cluster.hierarchy import linkage, fcluster


def all_sum_item(item):
    item = torch.tensor(item).cuda()
    dist.all_reduce(item)
    return item.item()

def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])
   
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind])
    total_instances = y_pred.size
    try: 
        if dist.get_world_size() > 0:
            total_acc = all_sum_item(total_acc)
            total_instances = all_sum_item(total_instances)
    except:
        pass
    total_acc /= total_instances

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    
    try:
        if dist.get_world_size() > 0:
            old_acc = all_sum_item(old_acc)
            total_old_instances = all_sum_item(total_old_instances)
    except:
        pass
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    
    try:
        if dist.get_world_size() > 0:
            new_acc = all_sum_item(new_acc)
            total_new_instances = all_sum_item(total_new_instances)
    except:
        pass
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc


def split_cluster_acc_v2_balanced(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}

    old_acc = np.zeros(len(old_classes_gt))
    total_old_instances = np.zeros(len(old_classes_gt))
    for idx, i in enumerate(old_classes_gt):
        old_acc[idx] += w[ind_map[i], i]
        total_old_instances[idx] += sum(w[:, i])

    new_acc = np.zeros(len(new_classes_gt))
    total_new_instances = np.zeros(len(new_classes_gt))
    for idx, i in enumerate(new_classes_gt):
        new_acc[idx] += w[ind_map[i], i]
        total_new_instances[idx] += sum(w[:, i])

    try:
        if dist.get_world_size() > 0:
            old_acc, new_acc = torch.from_numpy(old_acc).cuda(), torch.from_numpy(new_acc).cuda()
            dist.all_reduce(old_acc), dist.all_reduce(new_acc)
            dist.all_reduce(total_old_instances), dist.all_reduce(total_new_instances)
            old_acc, new_acc = old_acc.cpu().numpy(), new_acc.cpu().numpy()
            total_old_instances, total_new_instances = total_old_instances.cpu().numpy(), total_new_instances.cpu().numpy()
    except:
        pass

    total_acc = np.concatenate([old_acc, new_acc]) / np.concatenate([total_old_instances, total_new_instances])
    old_acc /= total_old_instances
    new_acc /= total_new_instances
    total_acc, old_acc, new_acc = total_acc.mean(), old_acc.mean(), new_acc.mean()
    return total_acc, old_acc, new_acc


EVAL_FUNCS = {
    'v2': split_cluster_acc_v2,
    'v2b': split_cluster_acc_v2_balanced
}

def log_accs_from_preds(y_true, y_pred, mask, eval_funcs, save_name, T=None,
                        print_output=True, args=None):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """
    if  isinstance(y_true,torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred,torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(mask,torch.Tensor):
        mask = mask.cpu().numpy()
    
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        all_acc, old_acc, new_acc = acc_f(y_true, y_pred, mask)
        log_name = f'{save_name}_{f_name}'

        if i == 0:
            to_return = (all_acc, old_acc, new_acc)

        if print_output:
            print_str = f'Epoch {T}, {log_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
            try:
                if dist.get_rank() == 0:
                    try:
                        args.logger.info(print_str)
                    except:
                        print(print_str)
            except:
                pass

    return to_return


def test_agglo(epoch, feats, targets, mask, save_name, args):
    best_acc = 0
    mask = mask.astype(bool)
    feats = np.concatenate(feats)
    linked = linkage(feats, method="ward")

    gt_dist = linked[:, 2][-args.num_labeled_classes-args.num_unlabeled_classes]
    preds = fcluster(linked, t=gt_dist, criterion='distance')
    test_all_acc_test, test_old_acc_test, test_new_acc_test = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name=save_name)

    dist = linked[:, 2][:-args.num_labeled_classes]
    tolerance = 0
    for d in reversed(dist):
        preds = fcluster(linked, t=d, criterion='distance')
        k = max(preds)
        all_acc_test, old_acc_test, new_acc_test = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                        T=0, eval_funcs=args.eval_funcs, save_name=save_name)

        if old_acc_test > best_acc:                                     # save best labeled acc without knowing GT K
            best_acc = old_acc_test
            best_acc_k = k
            best_acc_d = d
            tolerance = 0
        else:
            tolerance += 1
        
        if tolerance == 50 :
            break
    return test_all_acc_test, test_old_acc_test, test_new_acc_test, best_acc, best_acc_k

def test_linkage(epoch, feats, targets, mask, save_name, args):
    best_acc = 0
    mask = mask.astype(bool)
    feats = np.concatenate(feats)
    linked = linkage(feats, method="ward")

    gt_dist = linked[:, 2][-args.num_labeled_classes-args.num_unlabeled_classes]
  
    preds = fcluster(linked, t=gt_dist, criterion='distance')
    test_all_acc_test, test_old_acc_test, test_new_acc_test = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name=save_name)
    
    
    print_str = f'Epoch {epoch},Linkage: All {test_all_acc_test:.4f} | Old {test_old_acc_test:.4f} | New {test_new_acc_test:.4f}'
    #args.logger.info(print_str)
    return test_all_acc_test, test_old_acc_test, test_new_acc_test