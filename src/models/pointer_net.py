import torch
import torch.nn as nn


class PointerNet(nn.Module):
    def __init__(self,query_size,key_size,attention_type='affine'):
        super(PointerNet,self).__init__()

        assert attention_type in ('affine','dot')

        if attention_type == 'affine':
            self.key_linear = nn.Linear(key_size,query_size,bias=False)
        self.attention_type = attention_type
        self.input_linear = nn.Linear(query_size,query_size)
        self.type_linear = nn.Linear(32, query_size)

        self.V = nn.Parameter(torch.FloatTensor(query_size), requires_grad=True)
        nn.init.uniform_(self.V, -1, 1)
        self.tanh = nn.Tanh()
        self.context_linear = nn.Conv1d(key_size, query_size, 1, 1)
        self.coverage_linear = nn.Conv1d(1, query_size, 1, 1)
    def forward(self, key_encodings, query_encoding,key_mask=None):
        '''
        :param key_encodings: [B, len_q, dim]
        :param query_encodings: [tgt_num, B, dim]
        :param key_mask: [B, len_q]
        :return:
        '''
        if self.attention_type == 'affine':
            key_encodings = self.key_linear(key_encodings)
        #[B, 1, len_q, dim ]
        key_encodings = key_encodings.unsqueeze(1)
        #[B, tgt_num, dim, 1]
        q = query_encoding.permute(1,0,2).unsqueeze(3)
        #[B, tgt_num, len_q]
        weights = torch.matmul(key_encodings,q).squeeze(3)
        # (tgt_num, B, len_q)
        weights = weights.permute(1, 0, 2)

        if key_mask is not None:
            # (tgt_action_num, batch_size, src_sent_len)
            key_mask = key_mask.unsqueeze(0).expand_as(weights)
            weights.data.masked_fill_(key_mask.bool(), -float('inf'))

        return weights.squeeze(0)
