import time
import copy
from typing import List, Dict

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

def tensor_embeds(embed):
    return {k: torch.FloatTensor(embed[k]) for k in embed.index_to_key}

def get_word_embedding(embed: Dict[str, torch.Tensor], unk_rep: torch.Tensor, word: str) -> torch.Tensor:
    return embed[word] if word in embed else unk_rep

def get_paragraph_embedding(embed: Dict[str, torch.Tensor], unk_rep: torch.Tensor, words: List[str]) -> torch.Tensor:
    return torch.vstack([
        get_word_embedding(embed, unk_rep, word) for word in words
    ])

class AITADataset(Dataset):
    def __init__(self, posts, labels, embed: Dict[str, torch.Tensor], unk: torch.Tensor):
        super().__init__()
        self.embed = embed
        self.unk = unk
        self.posts = [word_tokenize(post.lower()) for post in posts]
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
          'post': get_paragraph_embedding(self.embed, self.unk, self.posts[idx]),
          'label': self.labels[idx]
        }


def basic_collate_fn(batch):
    """Collate function for basic setting."""
    posts = [i['post'] for i in batch]
    labels = torch.IntTensor([i['label'] for i in batch])

    return posts, labels

#######################################################################
################################ Models ###############################
#######################################################################

class BiGRU(nn.Module):
    def __init__(
        self,
        rnn_dense_hidden_dim: int,
        device: str,
        dropout_rate: float = 0.25,
        word_vec_length: int = 300
    ):
        super().__init__()
        self.device = device
        self.bigru = nn.GRU(
            word_vec_length,     # input is each word embedding
            word_vec_length,     # hidden representation is same size as word embeddings
            batch_first=False,
            bidirectional=True
        )
        self.bigruDropout1 = nn.Dropout(dropout_rate)
        self.bigruDense = nn.Linear(word_vec_length * 2, rnn_dense_hidden_dim)
        self.bigruDropout2 = nn.Dropout(dropout_rate)
        self.bigruOutput = nn.Linear(rnn_dense_hidden_dim, 1)
        
    def forward(self, posts):
        #### BiGRU recurrent layer ####################################################
        # posts shape: batch length * num_words_per_post (ragged) * w2vlen
        input_lengths = [seq.size(0) for seq in posts]
        padded_input = nn.utils.rnn.pad_sequence(posts) # tensor shape (max_post_len, batch_len, w2vlen)
        total_length = padded_input.size(0)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            padded_input, input_lengths, batch_first=False, enforce_sorted=False
        )
        packed_output, _ = self.bigru(packed_input) # shape (max_post_len, batch_len, rnn_hidden_dim)
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
        return self.bigruOutput(rnnDenseOut).flatten()
    
class textCNN(nn.Module):
    def __init__(
        self, 
        cnn_dense_hidden_dim: int,
        device: str,
        dropout_rate: float = 0.25,
        num_filters: int = 100,
        kernel_sizes: List[int] = [3, 4, 5],
        word_vec_length: int = 300
    ):
        super().__init__()
        # transpose input to get N * w2vlen * L, then  ===>  N * (num_filters x w2vlen) * L'
        self.convs = nn.ModuleList(
            [nn.Conv1d(word_vec_length, num_filters, k) for k in kernel_sizes]
        )
        # Compute max pooling along the L axis (not shown here), yielding N * (num_filters x w2vlen)
        self.cnnDropout1 = nn.Dropout(dropout_rate)
        self.cnnDense = nn.Linear(num_filters * len(kernel_sizes), cnn_dense_hidden_dim)
        self.cnnDropout2 = nn.Dropout(dropout_rate)
        self.cnnOutput = nn.Linear(cnn_dense_hidden_dim, 1)
        
    def forward(self, posts):
         #### Input reshaping ###########################################################
            # N * num_words_per_seq (ragged) * w2vlen ==> N * max_seq_len * w2vlen
        padded_input = nn.utils.rnn.pad_sequence(posts, batch_first=True) 
            # N * max_seq_len * w2vlen ==> N * w2vlen * max_seq_len (treat word2vec dimensions as channels)
        channelled_input = torch.transpose(padded_input, 1, 2)
        #### CNN clf convolutional layer ###############################################
            # Convolution: N * w2vlen * max_seq_len ==> N * num_filters * (max_seq_len - 2)
            # Pooling: N * num_filters * (max_seq_len - 2) ==> N * num_filters
        convOuts = [F.relu(conv(channelled_input)) for conv in self.convs]
        pooledOuts = [torch.max(convOut, dim=2).values for convOut in convOuts]
        pooledOut = torch.cat(pooledOuts, 1)
        pooledOut = self.cnnDropout1(pooledOut)
        #### CNN hidden layer ##########################################################
        cnnDenseOut = F.relu(self.cnnDense(pooledOut))
        cnnDenseOut = self.cnnDropout2(cnnDenseOut)
        #### CNN output layer ##########################################################
        return self.cnnOutput(cnnDenseOut).flatten()
        
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
        num_filters: int = 100,
        kernel_sizes: List[int] = [3, 4, 5],
        word_vec_length: int = 300
    ):
        super().__init__()
        self.device = device
        
        self.bigru = BiGRU(
            rnn_dense_hidden_dim, 
            device, 
            dropout_rate, 
            word_vec_length
        )
        self.cnn = textCNN(
            cnn_dense_hidden_dim, 
            device, 
            dropout_rate, 
            num_filters, 
            kernel_sizes, 
            word_vec_length
        )
        self.output = nn.Linear(2, 1)
    
    def forward(self, posts):
        cnn_out = self.cnn(posts)
        bigru_out = self.bigru(posts)
        combined_input = torch.stack((cnn_out, bigru_out), dim=1)
        
        predictionProbs = self.output(combined_input).flatten()
        
        return predictionProbs


#########################################################################
################################ Training ###############################
#########################################################################

def calculate_loss(scores, labels, loss_fn):
    if (scores.shape != labels.shape):
        print(scores)
        print(labels)
    return loss_fn(scores, labels.float())

def get_optimizer(net, lr, weight_decay):
    return optimizer.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)

def get_hyper_parameters():
    cnn_dense_hidden_dim = [256]
    rnn_dense_hidden_dim = [256]
    dropout_rate = [0.25, 0.5]
    lr = [3e-2, 3e-3, 3e-4]
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
        for posts, labels in trn_loader:
            num_itr += 1
            posts = [post.to(device) for post in posts]
            labels = labels.to(device)
            
            optim.zero_grad()
            output = net(posts)
            loss = calculate_loss(output, labels, loss_fn)
            loss.backward()
            optim.step()
            
            for name, param in net.named_parameters():
                print(name, param.grad)
            
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
    probs = torch.sigmoid(scores)
    return torch.IntTensor([1 if prob > 0.5 else 0 for prob in probs])

def get_validation_performance(net, loss_fn, data_loader, device):
    net.eval()
    y_true = [] # true labels
    y_pred = [] # predicted labels
    total_loss = [] # loss for each batch

    with torch.no_grad():
        for posts, labels in data_loader:
            posts = [post.to(device) for post in posts]
            labels = labels.to(device)
            loss = None # loss for this batch
            pred = None # predictions for this battch

            scores = net(posts)
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


def plot_loss(stats, display=True):
    """Plot training loss and validation loss."""
    plt.plot(stats['train_loss_ind'], stats['train_loss'], label='Training loss')
    plt.plot(stats['val_loss_ind'], stats['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    if display:
        plt.show()
    else:
        plt.savefig('best_nn_loss.png')

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

