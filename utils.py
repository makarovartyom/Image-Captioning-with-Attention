import string


def punctuation_free(reference):
    """Function takes a caption and outputs punctuation free and lower cased caption"""
    text = reference.split()
    x = [''.join(c.lower() for c in s if c not in string.punctuation) for s in text]
    return x


def get_batch_caps(caps_all, batch_size):
    """Function takes sampled captions for images and
    returns punctuation-free preprocessed caps in batches"""
    batch_caps_all = []
    for batch_idx in range(batch_size):
        batch_caps = [i for i in map(lambda t: (punctuation_free(t[batch_idx])), caps_all)]
        batch_caps_all.append(batch_caps)
    return batch_caps_all


def get_hypothesis(terms_idx, data_loader):
    """Function outputs word tokens from output indices (terms_idx)
    """
    hypothesis_list = []
    vocab = data_loader.dataset.vocab.idx2word
    for i in range(terms_idx.size(0)):
        words = [vocab.get(idx.item()) for idx in terms_idx[i]]
        words = [word for word in words if word not in (',', '.', '<end>')]
        hypothesis_list.append(words)
    return hypothesis_list
