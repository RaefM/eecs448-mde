import pandas as pd
import torch
import nltk
import itertools
from model import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm

"""
Requirements (see google drive):
    * glove300.kv, glove300.kv.npy
    * aita_minimal_preprocess.csv
    * nn/model.py
    
Saves best model it finds to best_ensemble_nn.pt
"""

def main():
    print("@@@@@@@@@@@@@@@@@@@@@\n@@@@ READING DATAFRAMES AND GLOVE\n@@@@@@@@@@@@@@@@@@@@@")
    aita_raw_df = pd.read_csv('aita_minimal_preprocess.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        aita_raw_df.post.values, 
        aita_raw_df.is_asshole.values,
        stratify = aita_raw_df.is_asshole.values,
        test_size = 0.2, 
        random_state = 448
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, 
        y_train,
        stratify = y_train,
        test_size = 0.2, 
        random_state = 448
    )
    
    embed = KeyedVectors.load('glove300.kv')
    unk = torch.FloatTensor(np.mean(embed.vectors, axis=0))
    embed_tensor_dict = tensor_embeds(embed)
    
    print("Done!")
    
    print("@@@@@@@@@@@@@@@@@@@@@\n@@@@ CREATING DATASETS AND LOADERS\n@@@@@@@@@@@@@@@@@@@@@")
    train_data = AITADataset(X_train, y_train, embed_tensor_dict, unk)
    dev_data = AITADataset(X_val, y_val, embed_tensor_dict, unk)
    test_data = AITADataset(X_test, y_test, embed_tensor_dict, unk)
    
    train_loader = DataLoader(train_data, batch_size=64, collate_fn=basic_collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=64, collate_fn=basic_collate_fn, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, collate_fn=basic_collate_fn, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Done! Operating on {device}!")
    
    print("@@@@@@@@@@@@@@@@@@@@@\n@@@@ EVALUATING HYPERPARAMETERS\n@@@@@@@@@@@@@@@@@@@@@")
    counts = [0, 0]
    for y in y_train:
        counts[y] += 1

    pos_weight = torch.Tensor([counts[0] / counts[1]]).to(device)
    
    print(f"Weighting positive samples {pos_weight} times more than negative ones")
    
    cnn_dense_hidden_dim, rnn_dense_hidden_dim, dropout_rate, lr, weight_decay = get_hyper_parameters()
    total_comb = len(cnn_dense_hidden_dim) * len(rnn_dense_hidden_dim) * len(dropout_rate) * len(lr) * len(weight_decay)
    print("CNN Hidden Size from: {}\n RNN Hidden Size from: {}\nLearning Rate from: {}\nWeight Decay from: {}\nDropout Rate from: {}".format(
        cnn_dense_hidden_dim, rnn_dense_hidden_dim, lr, weight_decay, dropout_rate
    ))
    best_model, best_stats = None, None
    best_accuracy, best_lr, best_wd, best_cnn_hd, best_rnn_hd, best_dr = 0, 0, 0, 0, 0, 0
    for cnn_hd, rnn_hd, dr, lr, wd in tqdm(
        itertools.product(cnn_dense_hidden_dim, rnn_dense_hidden_dim, dropout_rate, lr, weight_decay),
        total=total_comb
    ):
        net = ensembleCNNBiGRU(cnn_hd, rnn_hd, device, dr).to(device)
        optim = get_optimizer(net, lr=lr, weight_decay=wd)
        model, stats = train_model(net, train_loader, dev_loader, optim, num_epoch=100,
                                   collect_cycle=500, device=device, verbose=True, 
                                   patience=10, pos_weight=pos_weight)
        # print accuracy
        print(f"Completed {(cnn_hd, rnn_hd, dr, lr, wd)}: {stats['accuracy']}")
        # update best parameters if needed
        if stats['accuracy'] > best_accuracy:
            best_accuracy = stats['accuracy']
            best_model, best_stats = model, stats
            best_lr, best_wd, best_cnn_hd, best_rnn_hd, best_dr = lr, wd, cnn_hd, rnn_hd, dr
            torch.save(best_model.state_dict(), 'best_ensemble_nn.pt')
    print("\n\nBest cnn hidden: {}, Best rnn hidden: {}, Best weight_decay: {}, Best lr: {}, Best dropout: {}".format(
        best_cnn_hd, best_rnn_hd, best_wd, best_lr, best_dr))
    print("Accuracy: {:.4f}".format(best_accuracy))
    plot_loss(best_stats, False)
    
    loss_fn = nn.BCEWithLogitsLoss()
    test_accuracy, test_total_loss = get_validation_performance(best_model, loss_fn, test_loader, device) 
    
    print(f"Accuracy on test set: {test_accuracy} with loss {test_total_loss}")

if __name__ == "__main__":
    main()