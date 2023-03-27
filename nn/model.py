import time
import copy
from typing import List

from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from gensim.models import KeyedVectors
import torch
from torch.utils.data import Dataset
from torch import float32, nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
import gensim
import numpy as np
from sklearn.metrics import balanced_accuracy_score

def get_word_embedding(embed, unk_rep, word: str):
    return embed[word] if word in embed.key_to_index else unk_rep

def get_paragraph_embedding(embed, unk_rep, paragraph) -> ParagraphTensor:
    return torch.FloatTensor([
        get_word_embedding(embed, unk_rep, word) for word in word_tokenize(paragraph)
    ])

class AITADataset(Dataset):
    def __init__(self, posts, labels, embed, window_size=3):
        super().__init__()
        unk = np.mean(embed.vectors, axis=0)
        
        self.data = []
        for post, is_asshole in zip(posts, labels):
            self.data.append({
                'post': get_paragraph_embedding(post),
                'label': is_asshole
            })
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def basic_collate_fn(batch):
    """Collate function for basic setting."""
    posts = [i['post'] for i in batch]
    labels = torch.IntTensor([i['label'] for i in batch])

    return posts, labels

#######################################################################
################################ Model ################################
#######################################################################

class ensembleCNNBiGRU(nn.Module):
    """
    Based on:
    https://arxiv.org/pdf/1805.01890.pdf RMDL
    With code modelled off of:
    https://towardsdatascience.com/deep-learning-techniques-for-text-classification-78d9dc40bf7c
    
    A NN that assesses context (BiGRU feat) while not drowning 
    relevant phrases in said context (CNN feat)
    
    Computes max pooling over the time dimension (which represents
    ordering of words) for both feat representations before passing to
    shallow FFNN for final output
    
    Currently assumes that sentences and paragraphs are BOTH
    just series of words (instead of paragraphs being a series
    of sentences). This makes sense for the CNN as it is aiming
    to model only local dependencies anyway. For the BiGRU,
    it means we don't consider the relationships between sentences
    as a separate concept between that of words- in practice,
    I believe this shouldn't affect us- for example, sentence
    level relations include entailment/contradiction, coherence,
    consistency, etc, all of which feel secondary to meaning for us.
    In fact, we even consider punctuation currently so that's fine too.
    """
    def __init__(
        self, 
        cnn_dense_hidden_dim: int,
        rnn_dense_hidden_dim: int,
        device: str,
        dropout_rate: float = 0.25,
        num_filters: int = 32,
        kernel_size: int = 3,
        word_vec_length: int = 300
    ):
        super().__init__()
        self.device = device
        
        # CNN ARCHITECTURE ##############################################################################
        # transpose input to get N * w2vlen * L, then  ===>  N * (num_filters x w2vlen) * L'
        self.conv = nn.Conv1d(word_vec_length, num_filters, kernel_size)
        # Compute max pooling along the L axis (not shown here), yielding N * (num_filters x w2vlen)
        self.cnnDropout1 = nn.Dropout(dropout_rate)
        self.cnnDense = nn.Linear(num_filters * word_vec_length, cnn_dense_hidden_dim)
        self.cnnDropout2 = nn.Dropout(dropout_rate)
        self.cnnOutput = nn.Linear(cnn_dense_hidden_dim, 1)
        ################################################################################################
        
        # BiGRU ARCHITECTURE ###########################################################################
        self.BiGRU = nn.GRU(
            word_vec_length,     # input is each word embedding
            word_vec_length,     # hidden representation is same size as word embeddings
            batch_first=False,
            bidirectional=True
        )
        self.bigruDropout1 = nn.Dropout(dropout_rate)
        self.bigruDense = nn.Linear(word_vec_length * 2, rnn_dense_hidden_dim)
        self.bigruDropout2 = nn.Dropout(dropout_rate)
        self.bigruOutput = nn.Linear(rnn_dense_hidden_dim, 1)
        ################################################################################################
        
        # Output Layer #################################################################################
        self.output = nn.Linear(cnn_dense_hidden_dim + rnn_dense_hidden_dim, 1)
        ################################################################################################
    
    def forward(self, posts: List[ParagraphTensor]):
        def cnnForward(l_of_seqs):
            #### Input reshaping ###########################################################
                # N * num_words_per_seq (ragged) * wordveclen ==> N * max_seq_len * w2vlen
            padded_input = nn.utils.rnn.pad_sequence(l_of_seqs, batch_first=True) 
                # N * max_seq_len * w2vlen ==> N * w2vlen * max_seq_len (treat word2vec dimensions as channels)
            channelled_input = torch.transpose(padded_input, 1, 2)
            #### CNN clf convolutional layer ###############################################
                # Convolution: N * w2vlen * max_seq_len ==> N * num_filters * (max_seq_len - 2)
            convOut = F.relu(self.conv(channelled_input))
                # Pooling: N * num_filters * (max_seq_len - 2) ==> N * num_filters
            pooledOut = torch.max(convOut, dim=2).values
            pooledOut = self.cnnDropout1(pooledOut)
            #### CNN hidden layer ##########################################################
            cnnDenseOut = F.relu(self.cnnDense(pooledOut))
            cnnDenseOut = self.cnnDropout2(cnnDenseOut1)
            #### CNN output layer ##########################################################
            return F.sigmoid(self.cnnOutput(cnnDenseOut1))
        
        def biGRUForward(l_of_seqs):
            #### BiGRU recurrent layer ####################################################
            # l_of_seqs shape: batch length * num_words_per_seq (ragged) * w2vlen
            input_lengths = [seq.size(0) for seq in l_of_seqs]
            padded_input = nn.utils.rnn.pad_sequence(l_of_seqs) # tensor shape (max_seq_len, batch_len, w2vlen)
            total_length = padded_input.size(0)
            packed_input = nn.utils.rnn.pack_padded_sequence(
                padded_input, input_lengths, batch_first=False, enforce_sorted=False
            )
            packed_output, _ = self.BiGRU(packed_input) # shape (max_seq_len, batch_len, rnn_hidden_dim)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=False, total_length=total_length
            )
            # compute max pooling along the time dimension to collapse into a single rnn_hidden_dim vector
            rnn_embeddings = torch.max(output, dim=0).values
            rnn_embeddings = self.bigruDropout1(rnn_embeddings)
            #### BiGRU hidden layer #######################################################
            rnnDenseOut = F.relu(self.bigruDense(rnn_embeddings))
            rnnDenseOut = self.bigruDropout2(rnnDenseOut)
            ### BiGRU output layer ########################################################
            return F.sigmoid(self.bigruOutput(rnn_embeddings))
            
        cnn_embeds = cnnForward(posts)    # N * cnn_dense_hidden_dim
        rnn_embeds = biGRUForward(posts)  # N * rnn_dense_hidden_dim
        
        combined_input = torch.vstack((cnn_embeds, rnn_embeds))
        predictionProbs = F.sigmoid(self.output(combined_input))
        
        return predictionProbs


#########################################################################
################################ Training ###############################
#########################################################################

def calculate_loss(scores, labels, loss_fn):
    return loss_fn(scores, labels.float())

def get_optimizer(net, lr, weight_decay):
    return optimizer.Adam(net.parameters(), lr=lr)

def get_hyper_parameters():
    cnn_dense_hidden_dim = [1024, 2048, 4096]
    rnn_dense_hidden_dim = [256, 512]
    dropout_rate = [0.25, 0.5]
    lr = [3e-3, 3e-4]
    weight_decay = [0, 0.01]
    
    return cnn_dense_hidden_dim, rnn_dense_hidden_dim, dropout_rate, lr, weight_decay


def train_model(net, trn_loader, val_loader, optim, num_epoch=50, collect_cycle=30,
        device='cpu', verbose=True, patience=8, stopping_criteria='loss'):
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_accuracy = None, 0

    loss_fn = nn.BCEWithLogitsLoss()
    early_stopper = EarlyStopperLoss(patience) if stopping_criteria == 'loss' else EarlyStopperAcc(patience)
    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in range(num_epoch):
        # Training:
        net.train()
        for windows, labels in trn_loader:
            num_itr += 1
            windows = [[s.to(device) for s in window] for window in windows]
            labels = labels.to(device)
            
            optim.zero_grad()
            output = net(windows)
            loss = calculate_loss(output, labels, loss_fn)
            loss.backward()
            optim.step()
            
            if num_itr % collect_cycle == 0:  # Data collection cycle
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item()
                ))

        # Validation:
        accuracy, loss = get_validation_performance(net, loss_fn, val_loader, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation accuracy: {:.4f}".format(accuracy))
            print("Validation loss: {:.4f}".format(loss))
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(net)
            best_accuracy = accuracy
        if patience is not None and early_stopper.early_stop(
            loss if stopping_criteria == 'loss' else accuracy
        ):
            break
    
    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_loss_ind': val_loss_ind,
             'accuracy': best_accuracy,
    }

    return best_model, stats


def get_predictions(scores: torch.Tensor):
    return torch.IntTensor([1 if score > 0 else 0 for score in scores])

def get_validation_performance(net, loss_fn, data_loader, device):
    net.eval()
    y_true = [] # true labels
    y_pred = [] # predicted labels
    total_loss = [] # loss for each batch

    with torch.no_grad():
        for windows, labels in data_loader:
            windows = [[s.to(device) for s in window] for window in windows]
            labels = labels.to(device)
            loss = None # loss for this batch
            pred = None # predictions for this battch

            scores = net(windows)
            loss = calculate_loss(scores, labels, loss_fn)
            pred = torch.IntTensor(get_predictions(scores)).to(device)

            total_loss.append(loss.item())
            y_true.append(labels.cpu())
            y_pred.append(pred.cpu())
    
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    accuracy = balanced_accuracy_score(y_true, y_pred)
    total_loss = sum(total_loss) / len(total_loss)
    
    return accuracy, total_loss


def plot_loss(stats):
    """Plot training loss and validation loss."""
    plt.plot(stats['train_loss_ind'], stats['train_loss'], label='Training loss')
    plt.plot(stats['val_loss_ind'], stats['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.show()

class EarlyStopperAcc:
    def __init__(self, patience=5):
        self.patience = patience
        self.iters_below = 0
        self.iters_staying_same = 0
        self.max_acc = -float("inf")

    def early_stop(self, curr_acc):
        if curr_acc > self.max_acc:
            self.max_acc = curr_acc
            self.iters_below = 0
            self.iters_staying_same = 0
        elif curr_acc == self.max_acc:
            self.iters_staying_same += 1
            if self.iters_staying_same >= self.patience * 10:
                return True
        elif curr_loss < self.min_loss:
            self.iters_below += 1
            self.iters_staying_same += 1
            if self.iters_below >= self.patience or self.iters_staying_same >= self.patience * 10:
                return True
        return False

class EarlyStopperLoss:
    # Code inspired from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch u/isle_of_gods
    def __init__(self, patience=10):
        self.patience = patience
        self.iters_since_last_dec = 0
        self.min_loss = float("inf")

    def early_stop(self, curr_loss):
        if curr_loss < self.min_loss:
            self.min_loss = curr_loss
            self.iters_since_last_dec = 0
        elif curr_loss >= self.min_loss:
            self.iters_since_last_dec += 1
            if self.iters_since_last_dec >= self.patience:
                return True
        return False

