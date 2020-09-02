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

class GetStats:
    """Class for getting statistics ffrom text training/validation files"""
    def __init__(self, log_train_path, log_vali_path, bleu_path):
        self.log_train_path = log_train_path
        self.log_valid_path = log_valid_path
        self.bleu_path = bleu_path
        assert log_train_path == 'training_log.txt', "file must contain training logs and have a name 'training_log.txt'"
        assert log_valid_path == 'validation_log.txt', "file must contain validation logs and a name 'validation_log.txt'"
        assert bleu_path == 'bleu.txt', "file must contain bleu scores and have a name 'bleu.txt'"

        train_file = open(log_train_path, "r")
        train_logs = train_file.readlines()
        train_file.close()

        valid_file = open(log_valid_path, "r")
        valid_logs = valid_file.readlines()
        valid_file.close()

        bleu_file = open("bleu.txt", "r")
        bleu_score = bleu_file.readlines()
        bleu_file.close()

    def get_train_loss(self):
        """Returns training log from training_log.txt file"""
        losses=[]
        perplex=[]
        #assert self.log_file_path == 'training_log.txt', "Outputs training log only. For vailidation loss or BLEU score use get_valid_loss and get_bleu."
        for line in self.train_logs:
            loss = re.search('Loss train: (.*), Perplexity train:', line).group(1)
            losses.append(loss)
            perp = re.search('Perplexity train: (.*)\n', line).group(1)
            perplex.append(perp)
        return lossses, perplex

    def get_valid_log(self):
        """Returns validation log from validation_log.txt file"""
        losses = []
        perplex =[]
        for line in self.valid_logs:
            loss = re.search('Loss valid: (.*), Perplexity valid:', line).group(1)
            losses.append(loss)
            perp = re.search('Perplexity valid: (.*)\n', line).group(1)
            perplex.append(perp)
        return losses, perplex

    def get_bleu(self):
        """Returns BLEU scores from the text file"""
        bleu_1_scores=[]
        bleu_2_scores=[]
        bleu_3_scores=[]
        bleu_4_scores=[]

        for line in self.bleu_score:
            bleu_1 = re.search('BLEU-1: (.*), BLEU-2:', line).group(1)
            bleu_1_scores.append(bleu_1)

            bleu_2 = re.search('BLEU-2: (.*), BLEU-3:', line).group(1)
            bleu_2_scores.append(bleu_2)

            bleu_3 = re.search('BLEU-3: (.*), BLEU-4:', line).group(1)
            bleu_3_scores.append(bleu_3)

            bleu_4 = re.search('BLEU-4: (.*)\n', line).group(1)
            bleu_4_scores.append(bleu_4)
        return bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores
