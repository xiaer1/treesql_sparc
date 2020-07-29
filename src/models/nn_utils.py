import torch
import torch.nn.functional as F

def identity(x):
    return x

def linear_layer(exp, weights, biases=None):
    # exp: input as size_1 or 1 x size_1
    # weight: size_1 x size_2
    # bias: size_2
    if exp.dim() == 1:
        exp = torch.unsqueeze(exp, 0)
    assert exp.size()[1] == weights.size()[0]
    if biases is not None:
        assert weights.size()[1] == biases.size()[0]
        result = torch.mm(exp, weights) + biases
    else:
        result = torch.mm(exp, weights)
    return result

def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
    """
    :param h_t: (batch_size, hidden_size)
    :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
    :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
    :param mask: (batch_size, src_sent_len)
    """
    # (batch_size, src_sent_len)
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask is not None:
        att_weight.data.masked_fill_(mask, -float('inf'))
    #TODO （B,src_sent_len) 每个句子的 每个词 的权重
    att_weight = F.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    # (batch_size, hidden_size)
    #TODO 将句子的每个词 作 加权得到每句话的 表示 ， 句子向量
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

    return ctx_vec, att_weight