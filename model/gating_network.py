import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class HGN(nn.Module):
    def __init__(self, num_users, num_items, model_args, device):
        super(HGN, self).__init__()

        self.args = model_args

        # init args
        L = self.args.L
        dims = self.args.d

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims).to(device)
        self.item_embeddings = nn.Embedding(num_items, dims).to(device)

        self.feature_gate_item = nn.Linear(dims, dims).to(device)
        self.feature_gate_user = nn.Linear(dims, dims).to(device)

        self.instance_gate_item = Variable(torch.zeros(dims, 1).type(torch.FloatTensor), requires_grad=True).to(device)
        self.instance_gate_user = Variable(torch.zeros(dims, L).type(torch.FloatTensor), requires_grad=True).to(device)
        self.instance_gate_item = torch.nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_user = torch.nn.init.xavier_uniform_(self.instance_gate_user)

        self.W2 = nn.Embedding(num_items, dims, padding_idx=0).to(device)
        self.b2 = nn.Embedding(num_items, 1, padding_idx=0).to(device)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self, item_seq, user_ids, items_to_predict, for_pred=False):
        item_embs = self.item_embeddings(item_seq)
        user_emb = self.user_embeddings(user_ids)

        # feature gating
        gate = torch.sigmoid(self.feature_gate_item(item_embs) + self.feature_gate_user(user_emb).unsqueeze(1))
        gated_item = item_embs * gate

        # instance gating
        instance_score = torch.sigmoid(torch.matmul(gated_item, self.instance_gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.instance_gate_user))
        union_out = gated_item * instance_score.unsqueeze(2)
        union_out = torch.sum(union_out, dim=1)
        union_out = union_out / torch.sum(instance_score, dim=1).unsqueeze(1)

        w2 = self.W2(items_to_predict)
        b2 = self.b2(items_to_predict)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()

            # MF
            res = user_emb.mm(w2.t()) + b2

            # union-level
            res += union_out.mm(w2.t())

            # item-item product
            rel_score = torch.matmul(item_embs, w2.t().unsqueeze(0))
            rel_score = torch.sum(rel_score, dim=1)
            res += rel_score
        else:
            # MF
            res = torch.baddbmm(b2, w2, user_emb.unsqueeze(2)).squeeze()

            # union-level
            res += torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()

            # item-item product
            rel_score = item_embs.bmm(w2.permute(0, 2, 1))
            rel_score = torch.sum(rel_score, dim=1)
            res += rel_score

        return res