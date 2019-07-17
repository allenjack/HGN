from model.gating_network import HGN
from interactions import Interactions
from eval_metrics import *

import argparse
import logging
from time import time
import datetime
import torch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluation(hgn, train, test_set, topk=20):
    num_users = train.num_users
    num_items = train.num_items
    batch_size = 1024
    num_batches = int(num_users / batch_size) + 1
    user_indexes = np.arange(num_users)
    item_indexes = np.arange(num_items)
    pred_list = None
    train_matrix = train.tocsr()
    test_sequences = train.test_sequences.sequences

    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index = user_indexes[start:end]

        batch_test_sequences = test_sequences[batch_user_index]
        batch_test_sequences = np.atleast_2d(batch_test_sequences)

        batch_test_sequences = torch.from_numpy(batch_test_sequences).type(torch.LongTensor).to(device)
        item_ids = torch.from_numpy(item_indexes).type(torch.LongTensor).to(device)
        batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(device)

        rating_pred = hgn(batch_test_sequences, batch_user_ids, item_ids, True)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0

        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(rating_pred, -topk)
        ind = ind[:, -topk:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if batchID == 0:
            pred_list = batch_pred_list
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)

    precision, recall, MAP, ndcg = [], [], [], []
    for k in [5, 10, 15, 20]:
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))
        MAP.append(mapk(test_set, pred_list, k))
        ndcg.append(ndcg_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg


def negsamp_vectorized_bsearch_preverif(pos_inds, n_items, n_samp=32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    reference: https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds


def generate_negative_samples(train_matrix, num_neg=3, num_sets=10):
    neg_samples = []
    for user_id, row in enumerate(train_matrix):
        pos_ind = row.indices
        neg_sample = negsamp_vectorized_bsearch_preverif(pos_ind, train_matrix.shape[1], num_neg * num_sets)
        neg_samples.append(neg_sample)

    return np.asarray(neg_samples).reshape(num_sets, train_matrix.shape[0], num_neg)


def train_model(train_data, test_data, config):
    num_users = train_data.num_users
    num_items = train_data.num_items

    # convert to sequences, targets and users
    sequences_np = train_data.sequences.sequences
    targets_np = train_data.sequences.targets
    users_np = train_data.sequences.user_ids
    train_matrix = train_data.tocsr()

    n_train = sequences_np.shape[0]
    logger.info("Total training records:{}".format(n_train))

    hgn = HGN(num_users, num_items, config, device).to(device)

    optimizer = torch.optim.Adam(hgn.parameters(), lr=config.learning_rate, weight_decay=config.l2)

    record_indexes = np.arange(n_train)
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1
    for epoch_num in range(config.n_iter):

        t1 = time()

        # set model to training mode
        hgn.train()

        np.random.shuffle(record_indexes)

        t_neg_start = time()
        negatives_np_multi = generate_negative_samples(train_matrix, config.neg_samples, config.sets_of_neg_samples)
        logger.info("Negative sampling time: {}s".format(time() - t_neg_start))

        epoch_loss = 0.0
        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break

            batch_record_index = record_indexes[start:end]

            batch_users = users_np[batch_record_index]
            batch_sequences = sequences_np[batch_record_index]
            batch_targets = targets_np[batch_record_index]
            negatives_np = negatives_np_multi[batchID % config.sets_of_neg_samples]
            batch_neg = negatives_np[batch_users]

            batch_users = torch.from_numpy(batch_users).type(torch.LongTensor).to(device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(device)
            batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(device)
            batch_negatives = torch.from_numpy(batch_neg).type(torch.LongTensor).to(device)

            items_to_predict = torch.cat((batch_targets, batch_negatives), 1)
            prediction_score = hgn(batch_sequences, batch_users, items_to_predict, False)

            (targets_prediction, negatives_prediction) = torch.split(
                prediction_score, [batch_targets.size(1), batch_negatives.size(1)], dim=1)

            # compute the BPR loss
            loss = -torch.log(torch.sigmoid(targets_prediction - negatives_prediction) + 1e-8)
            loss = torch.mean(torch.sum(loss))

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= num_batches

        t2 = time()

        output_str = "Epoch %d [%.1f s]  loss=%.4f" % (epoch_num + 1, t2 - t1, epoch_loss)
        logger.info(output_str)

        if (epoch_num + 1) % 20 == 0:
            hgn.eval()
            precision, recall, MAP, ndcg = evaluation(hgn, train_data, test_data, topk=20)
            logger.info(', '.join(str(e) for e in precision))
            logger.info(', '.join(str(e) for e in recall))
            logger.info(', '.join(str(e) for e in MAP))
            logger.info(', '.join(str(e) for e in ndcg))
            logger.info("Evaluation time:{}".format(time() - t2))
    logger.info("\n")
    logger.info("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=3)

    # train arguments
    parser.add_argument('--n_iter', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--sets_of_neg_samples', type=int, default=50)

    # model dependent arguments
    parser.add_argument('--d', type=int, default=50)

    config = parser.parse_args()

    from data import Amazon
    data_set = Amazon.CDs()  # MovieLens.ML20M()  # Books, CDs, Electronics
    # item_id=0 for sequence padding
    train_set, val_set, train_val_set, test_set, num_users, num_items = data_set.generate_dataset(index_shift=1)
    train = Interactions(train_val_set, num_users, num_items)
    train.to_sequence(config.L, config.T)

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(config)
    train_model(train, test_set, config)
