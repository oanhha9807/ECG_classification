# Code based on the source code of homework 1 and homework 2 of the
# deep structured learning code https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks
from tqdm import tqdm
import argparse
import torch
from mne.viz import plot_epochs_image
from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import configure_seed, configure_device, compute_scores, Dataset_for_RNN, \
    plot_losses
from datetime import datetime
import statistics
import numpy as np
import os
from sklearn.metrics import roc_curve
from torchmetrics.classification import MultilabelAUROC
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss




from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MLP Class to process multi-hot vector



# Label Correlation Embedding (for learning label embedding space)
class LabelEmbedding(nn.Module):
    def __init__(self, num_labels, embedding_dim):
        super(LabelEmbedding, self).__init__()
        self.embedding = nn.Embedding( num_labels, embedding_dim)

    def forward(self):
        return self.embedding.weight  # Returns the embeddings for all labels


# Cosine Similarity function
def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=1)


# GCN Layer to propagate label correlations
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(input_dim, output_dim)

    def forward(self, x, edge_index):
        return F.relu(self.conv(x, edge_index))




def create_graph_from_correlation0(correlation_matrix, threshold=0.6):
# def create_graph_from_correlation(correlation_matrix, threshold=0.5):
    num_channels = correlation_matrix.shape[0]
    edge_index = []
    edge_attr = []


    for i in range(num_channels):
        for j in range(num_channels):

            if i != j and (correlation_matrix[i, j] > threshold ):#best
                value = sorted([i, j])
                cosines = cosine(data[:, id_1], data[:, id_2])
                edge_index.append(value)
                # edge_attr.append([correlation_matrix[i, j]])


    edge_index = np.asarray(edge_index)

    edge_index = np.unique(edge_index, axis = 0)
    tmp = edge_index

    for idx in edge_index:
        m, n = idx
        edge_attr.append([correlation_matrix[m,n]])
    edge_index = list(edge_index)
    # edge_attr = np.asarray(edge_attr)
    if tmp.shape[0] == 0:
        edge_index.append([0,0])
        edge_attr.append([0.0])
    edge_index = np.asarray(edge_index)
    edge_attr = np.asarray(edge_attr)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr



def get_edge_index_and_weights_from_correlation(correlation_matrices, threshold=0.5):

    batch_size = correlation_matrices.size(0)
    num_labels = correlation_matrices.size(1)  # Should be 4 for your case

    edge_indices_list = []
    edge_weights_list = []

    for i in range(batch_size):
        # Get the current correlation matrix for the sample
        correlation_matrix = correlation_matrices[i]
        # print(correlation_matrix)

        edge_indices, edge_weights =  create_graph_from_correlation0(correlation_matrix, threshold = 0.2)
        # print( edge_indices, edge_weights)
        # Store the edge indices and weights
        edge_indices_list.append(edge_indices)
        edge_weights_list.append(edge_weights)


    edge_index = torch.cat(edge_indices_list, dim=1)
    edge_weight = torch.cat(edge_weights_list, dim=0)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().cuda()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float).cuda()

    return edge_index, edge_weight


def compute_correlation_matrix(batch_data):
    """
    Compute correlation matrix for each sample in the batch.

    Args:
        batch_data (torch.Tensor): Tensor of shape [B, 4], where B is batch size and 4 is label size.

    Returns:
        torch.Tensor: Tensor of shape [B, 4, 4], containing correlation matrices for each sample.
    """
    # Ensure input is a PyTorch tensor
    if not isinstance(batch_data, torch.Tensor):
        batch_data = torch.tensor(batch_data, dtype=torch.float32)

    # Normalize each vector to compute cosine similarity
    norms = torch.norm(batch_data, dim=1, keepdim=True) + 1e-8
    normalized_batch = batch_data / norms

    # Compute outer product for each sample
    batch_size, num_labels = batch_data.size()
    correlation_matrices = torch.zeros(batch_size, num_labels, num_labels)

    for i in range(batch_size):
        sample = normalized_batch[i].view(-1, 1)  # [4, 1]
        correlation_matrices[i] = torch.mm(sample, sample.T)  # [4, 4]

    return correlation_matrices


class LabelCorrelationGCN(nn.Module):
    def __init__(self, num_labels, embedding_dim):
        super(LabelCorrelationGCN, self).__init__()
        self.gcn1 = GCNConv(num_labels, embedding_dim)
        self.gcn2 = GCNConv(embedding_dim, embedding_dim)

    def forward(self, label_vectors):
        # print(label_vectors.shape)
        # Calculate cosine similarity matrix
        correlation_matrices = compute_correlation_matrix(label_vectors)


        edge_index, edge_weight = get_edge_index_and_weights_from_correlation(correlation_matrices)

        # Apply GCN layers
        print(edge_index.shape, edge_weight.shape)
        edge_index = torch.permute(edge_index, (1, 0))

        x = self.gcn1(label_vectors, edge_index=edge_index, edge_weight = edge_weight)
        x = F.relu(x)
        x = self.gcn2(x, edge_index=edge_index, edge_weight = edge_weight)
        # print(x.shape)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Define the attention layer
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, rnn_output):
        # rnn_output shape: (batch_size, seq_length, hidden_size) if batch_first=True
        # rnn_output shape: (seq_length, batch_size, hidden_size) if batch_first=False
        if not self.batch_first:
            rnn_output = rnn_output.transpose(0, 1)  # (batch_size, seq_length, hidden_size)

        # Apply attention layer to the hidden states
        attn_weights = self.attention(rnn_output)  # (batch_size, seq_length, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Multiply the weights by the rnn_output to get a weighted sum
        context = torch.sum(attn_weights * rnn_output, dim=1)  # (batch_size, hidden_size)
        return context, attn_weights


class LabelEmbeddingMLP(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        """
        Multi-Layer Perceptron for label embedding.
        """
        super(LabelEmbeddingMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x):
        return self.mlp(x)
def compute_cosine_similarity(embeddings):
    """
    Compute the cosine similarity matrix from label embeddings.
    """
    normed_embeddings = F.normalize(embeddings, p=2, dim=1)
    cosine_similarity_matrix = torch.mm(normed_embeddings, normed_embeddings.t())
    return cosine_similarity_matrix

def compute_correlation_matrix(co_occurrence_counts, occurrence_counts):
    """
    Compute the conditional probability correlation matrix.
    """
    num_labels = len(occurrence_counts)
    correlation_matrix = np.zeros((num_labels, num_labels))

    for i in range(num_labels):
        for j in range(num_labels):
            if occurrence_counts[j] > 0:
                correlation_matrix[i, j] = co_occurrence_counts[i, j] / occurrence_counts[j]
    return correlation_matrix
def reweight_correlation_matrix(correlation_matrix, threshold, rescale_param):
    """
    Apply nonlinear reweighting to filter noise from the correlation matrix.
    """
    reweighted_matrix = np.maximum(0, (correlation_matrix - threshold) / rescale_param)
    return reweighted_matrix
def correlation_loss(predicted_similarity, target_correlation):
    """
    Loss function to minimize the difference between predicted similarity and target correlation.
    """
    return F.mse_loss(predicted_similarity, target_correlation)
# Định nghĩa GCN để học mối quan hệ giữa các lớp
class LabelGCN(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_dim):
        super(LabelGCN, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)  # Ánh xạ nhãn thành embedding
        self.mlp = LabelEmbeddingMLP(4, hidden_dim)  # MLP để học nhúng từ multi-hot vector
        self.gcn = GCNConv(4, embedding_dim)  # Áp dụng GCN để học mối quan hệ

    def forward(self, x):
        # Áp dụng MLP để học embedding cho các labels
        # compute co_occurrence_counts and occurrence_counts

        co_occurrence_counts = torch.mm(x.T, x)
        occurrence_counts = x.sum(dim=0)
        # Step 1: Build and reweight correlation matrix
        correlation_matrix = compute_correlation_matrix(co_occurrence_counts, occurrence_counts)
        reweighted_matrix = reweight_correlation_matrix(correlation_matrix, threshold=0.1, rescale_param=1.0)
        # print(correlation_matrix.shape)
        # print(correlation_matrix.shape)
        # Step 2: Convert to PyTorch tensor
        target_correlation = torch.tensor(reweighted_matrix, dtype=torch.float32)
        embeddings = self.mlp(x)
        # print(embeddings.shape)
        predicted_similarity = compute_cosine_similarity(embeddings)
        # print(predicted_similarity.shape, target_correlation.shape)
        # loss = correlation_loss(predicted_similarity, target_correlation)
        # print(loss)

        # label_embeddings = self.mlp(x)
        # # print(label_embeddings.shape)
        # edge_index, edge_weight = create_edge_index_and_weight(label_embeddings)
        #
        #
        # # Áp dụng GCN để học các mối quan hệ giữa các labels
        # x = self.gcn(label_embeddings, edge_index, edge_weight)
        return embeddings


# Hàm tính toán cosine similarity giữa hai vector
def cosine_similarity(e1, e2):
    return F.cosine_similarity(e1, e2, dim=-1)


# Hàm tạo edge_index và edge_weight từ cosine similarity
def create_edge_index_and_weight(label_embeddings):
    # num_labels = label_embeddings.size(1)  # Số lượng lớp (labels)
    num_labels = 4
    edge_index = []
    edge_weight = []

    # Tính toán mối quan hệ giữa các labels
    # print(label_embeddings)
    for i in range(num_labels):
        for j in range(i + 1, num_labels):
            sim = cosine_similarity(label_embeddings[i], label_embeddings[j])
            # print(sim)
            if sim > 0.25:  # Ngưỡng cosine similarity
                edge_index.append([i, j])
                edge_weight.append(sim.item())

    edge_index = torch.tensor(edge_index).t().contiguous().to(device)  # Chuyển thành tensor với dạng [2, num_edges]
    edge_weight = torch.tensor(edge_weight).to(device)   # Trọng số cho các edges
    return edge_index, edge_weight
class RNN_att(nn.Module):
    # ... [previous __init__ definition] ...

    def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout_rate, bidirectional, gpu_id=None):
        """
        Define the layers of the model
        Args:
            input_size (int): "Feature" size (in this case, it is 3)
            hidden_size (int): Number of hidden units
            num_layers (int): Number of hidden RNN layers
            n_classes (int): Number of classes in our classification problem
            dropout_rate (float): Dropout rate to be applied in all rnn layers except the last one
            bidirectional (bool): Boolean value: if true, gru layers are bidirectional
        """
        super(RNN_att, self).__init__()
        self.input_size = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.gpu_id = gpu_id
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=0, batch_first=True, bidirectional=bidirectional)  # batch_first: first dimension is the batch size
        self.rnn = nn.LSTM(1, 32, 2, dropout=0, batch_first=True, bidirectional=True)
        self.rnn1 = nn.LSTM(192, 128, 2, dropout=0, batch_first=True, bidirectional=True)
        # self.rnn2 = nn.LSTM(hidden_size*2, hidden_size, num_layers, dropout=dropout_rate, batch_first=True,
        #                    bidirectional=bidirectional)  # batch_first: first dimension is the batch size
        self.conv11 = nn.Conv1d(1000, 1000, kernel_size=3, padding= 1)
        if bidirectional:
            self.d = 2
        else:
            self.d = 1

        # Initialize the attention layer
        self.attention = Attention(256, batch_first=True)

        # Adjust the input dimension for the classification layer according to bidirectionality
        self.fc = nn.Linear(256, n_classes)
        num_classes = 4
        embedding_dim = 64

        hidden_dim = 64
        self.label_gcn = LabelGCN(4, embedding_dim, 192)
        self.classifier = nn.Linear(embedding_dim, num_classes)  # Dự đoán các lớp


    def forward(self, X, label_vectors):

        list_f = []
        for i in range(3):
            X_tmp = X[:,:,i:i+1]
            # print(X_tmp.shape)
            # print(X.shape)
            out_rnn, _ = self.rnn(X_tmp)
            # out_rnn, _ = self.rnn2(out_rnn)
            list_f.append(out_rnn)
            # print(out_rnn.shape)
        # print(list_f.shape)
        out_rnn = torch.cat(list_f, dim=2)
        # print(out_rnn.shape)
        # attn_output shape: (batch_size, hidden_size*d)
        label_embeddings = self.label_gcn(label_vectors)
        # print(out_rnn.shape)#128, 1000, 192
        # print(label_embeddings.shape)#128, 1x192
        label_embeddings = label_embeddings[:, None, :]
        elementwise_product = out_rnn * label_embeddings
        f_norm = torch.norm(out_rnn, dim=1, keepdim=True)  # Norm for each row in f (1000, 1)
        e_norm = torch.norm(label_embeddings)  # Single scalar norm for e
        fused_feature = F.relu(elementwise_product / (f_norm * e_norm))
        fused_feature = self.conv11(fused_feature) + out_rnn
        # tmp = torch.cat((out_rnn, label_embeddings), dim=1)
        tmp, _ = self.rnn1(fused_feature)
        # print(tmp.shape)
        attn_output, attn_weights = self.attention(tmp)

        # Multi-label predictions


        logits = self.fc(attn_output)
        # print(logits)
        # out_fc shape: (batch_size, n_classes)
        # print(logits.shape)

        return logits


    def label_loss(self, preds, labels, cosine_sim):
        # Binary Cross Entropy Loss for multi-label classification
        bce_loss = F.binary_cross_entropy_with_logits(preds, labels)

        # Cosine Similarity Loss (penalize less similar labels)
        cosine_loss = torch.mean(1 - cosine_sim)  # Maximizing similarity

        # Final loss combining BCE and cosine similarity loss
        total_loss = bce_loss + self.lambda_cosine * cosine_loss
        return total_loss
def train_batch(X, y, model, optimizer, gpu_id=None, **kwargs):
    """
    X (batch_size, 1000, 3): batch of examples
    y (batch_size, 4): ground truth labels_train
    model: Pytorch model
    optimizer: optimizer for the gradient step
    criterion: loss function
    """
    X, y = X.to(gpu_id), y.to(gpu_id)
    optimizer.zero_grad()
    out = model(X,y)

    # loss = criterion(out, y)
    loss = sigmoid_focal_loss(out, y, alpha=0.5, gamma=1.0, reduction='mean')
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X, y, thr):
    """
    Make labels_train predictions for "X" (batch_size, 1000, 3)
    """
    x_batch, y_batch = X.to(device), y.to(device)
    # label_embeddings = mlp(y_batch)
    # edge_index, edge_weight = create_edge_index_and_weight(label_embeddings)
    # edge_index = edge_index.to(device)
    # edge_weight = edge_weight.to(device)
    logits_ = model(x_batch, y_batch)
    # logits_ = model(X, y)  # (batch_size, n_classes)
    probabilities = torch.sigmoid(logits_).cpu()

    if thr is None:
        return probabilities
    else:
        return np.array(probabilities.numpy() >= thr, dtype=float)


def evaluate(model, dataloader, thr, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size,4): ground truth labels_train
    """
    model.eval()  # set dropout and batch normalization layers to evaluation mode
    with torch.no_grad():
        matrix = np.zeros((4, 4))
        for i, (x_batch, y_batch) in tqdm(enumerate(dataloader)):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            y_pred = predict(model,x_batch,y_batch, thr)
            y_true = np.array(y_batch.cpu())
            matrix = compute_scores(y_true, y_pred, matrix)

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

    return matrix
    # cols: TP, FN, FP, TN


def auroc(model, dataloader, gpu_id=None):
    """
    model: Pytorch model
    X (batch_size, 1000, 3) : batch of examples
    y (batch_size,4): ground truth labels_train
    """
    model.eval()  # set dropout and batch normalization layers to evaluation mode
    with torch.no_grad():
        preds = []
        trues = []
        for i, (x_batch, y_batch) in tqdm(enumerate(dataloader)):
            # print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)

            preds += predict(model, x_batch, None)
            trues += [y_batch.cpu()[0]]

            del x_batch
            del y_batch
            torch.cuda.empty_cache()

    preds = torch.stack(preds)
    trues = torch.stack(trues).int()
    return MultilabelAUROC(num_labels=4, average=None)(preds, trues)
    # cols: TP, FN, FP, TN


# Validation loss
def compute_loss(model, dataloader, gpu_id=None):
    model.eval()
    with torch.no_grad():
        val_losses = []
        # mlp = MLP(4, 32, 32).to(device)
        for i, (x_batch, y_batch) in tqdm(enumerate(dataloader)):
            # print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            # label_embeddings = mlp(y_batch)
            # edge_index, edge_weight = create_edge_index_and_weight(label_embeddings)
            # edge_index = edge_index.to(device)
            # edge_weight = edge_weight.to(device)
            y_pred = model(x_batch, y_batch)
            # loss = criterion(y_pred, y_batch)
            loss = sigmoid_focal_loss(y_pred, y_batch, alpha=0.5, gamma=1.0, reduction='mean')
            val_losses.append(loss.item())
            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

        return statistics.mean(val_losses)


def threshold_optimization(model, dataloader, gpu_id=None):
    """
    Make labels_train predictions for "X" (batch_size, 1000, 3)
    """
    save_probs = []
    save_y = []
    threshold_opt = np.zeros(4)

    model.eval()
    with torch.no_grad():
        #threshold_opt = np.zeros(4)
        for _, (X, Y) in tqdm(enumerate(dataloader)):
            X, Y = X.to(gpu_id), Y.to(gpu_id)
            # mlp = MLP(4, 32, 32)
            # label_embeddings = mlp(Y)
            # edge_index, edge_weight = create_edge_index_and_weight(label_embeddings)
            # edge_index = edge_index.to(device)
            # edge_weight = edge_weight.to(device)

            logits_ = model(X, Y)  # (batch_size, n_classes)
            probabilities = torch.sigmoid(logits_).cpu()
            Y = np.array(Y.cpu())
            save_probs += [probabilities.numpy()]
            save_y += [Y]

    # find the optimal threshold with ROC curve for each disease

    save_probs = np.array(np.concatenate(save_probs)).reshape((-1, 4))
    save_y = np.array(np.concatenate(save_y)).reshape((-1, 4))
    for dis in range(0, 4):
        # print(probabilities[:, dis])
        # print(Y[:, dis])
        fpr, tpr, thresholds = roc_curve(save_y[:, dis], save_probs[:, dis])
        # geometric mean of sensitivity and specificity
        gmean = np.sqrt(tpr * (1 - fpr))
        # optimal threshold
        index = np.argmax(gmean)
        threshold_opt[dis] = round(thresholds[index], ndigits=2)

    return threshold_opt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data',  default='/data/oanh/Bio/RNN_data/',
                        help="Path to the dataset.")
    parser.add_argument('-epochs', default=200, type=int,
                        help="""Number of epochs to train the model.""")
    parser.add_argument('-batch_size', default=128, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='adam')
    parser.add_argument('-gpu_id', type=int, default=0)
    parser.add_argument('-path_save_model', default='/home/oem/oanh/DL_ECG_Classification/save_models/lstm/',
                        help='Path to save the model')
    parser.add_argument('-num_layers', type=int, default=2)
    parser.add_argument('-hidden_size', type=int, default=64)#best = 64
    parser.add_argument('-bidirectional', type=bool, default=True)
    parser.add_argument('-early_stop', type=bool, default=False)
    parser.add_argument('-patience', type=int, default=20)
    opt = parser.parse_args()


    configure_seed(seed=42)
    configure_device(opt.gpu_id)

    samples = [17111, 2156, 2163]
    print("Loading data...")
    train_dataset = Dataset_for_RNN(opt.data, samples, 'train')
    dev_dataset = Dataset_for_RNN(opt.data, samples, 'dev')
    test_dataset = Dataset_for_RNN(opt.data, samples, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    dev_dataloader_thr = DataLoader(dev_dataset, batch_size=opt.batch_size, shuffle=False)

    input_size = 3
    hidden_size = opt.hidden_size
    num_layers = opt.num_layers
    n_classes = 4

    # initialize the model
    model = RNN_att(input_size, hidden_size, num_layers, n_classes, dropout_rate=opt.dropout, gpu_id=opt.gpu_id,
                bidirectional=opt.bidirectional)
    model = model.to(opt.gpu_id)

    # get an optimizer
    # optims = {
    #     "adam": torch.optim.Adam,
    #     "sgd": torch.optim.SGD}
    #
    # optim_cls = optims[opt.optimizer]
    # optimizer = optim_cls(
    #     model.parameters(),
    #     lr=opt.learning_rate,
    #     weight_decay=opt.l2_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)
    # get a loss criterion and compute the class weights (nbnegative/nbpositive)
    # according to the comments https://discuss.pytorch.org/t/weighted-binary-cross-entropy/51156/6
    # and https://discuss.pytorch.org/t/multi-label-multi-class-class-imbalance/37573/2
    class_weights = torch.tensor([17111/4389, 17111/3136, 17111/1915, 17111/417], dtype=torch.float)
    class_weights = class_weights.to(opt.gpu_id)
    # criterion = sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean')
    # criterion = nn.BCEWithLogitsLoss(
    #     pos_weight=class_weights)  # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_mean_losses = []
    train_losses = []
    epochs_run = opt.epochs
    max_acc = 0

    num_samples = 1000
    num_classes = 4
    embedding_dim = 16
    hidden_dim = 16



    # Tạo edge_index và edge_weight từ label_embeddings


    # mlp = MLP(num_classes, hidden_dim, embedding_dim).to(device)
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        print('Epoch-{0} lr: {1}'.format(ii, optimizer.param_groups[0]['lr']))
        for i, (X_batch, y_batch) in tqdm(enumerate(train_dataloader)):
            y_batch = y_batch.to(device)
            # label_embeddings = mlp(y_batch)
            # edge_index, edge_weight = create_edge_index_and_weight(label_embeddings)
            # edge_index = edge_index.to(device)
            # edge_weight = edge_weight.to(device)
            loss = train_batch(
                X_batch, y_batch, model, optimizer,gpu_id=opt.gpu_id)
            del X_batch
            del y_batch
            torch.cuda.empty_cache()
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        val_loss = compute_loss(model, dev_dataloader, gpu_id=opt.gpu_id)
        valid_mean_losses.append(val_loss)
        print('Validation loss: %.4f' % (val_loss))
        scheduler.step(val_loss)
        dt = datetime.now()
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # save the model at each epoch where the validation loss is the best so far
        if val_loss == np.min(valid_mean_losses):
            f = os.path.join(opt.path_save_model, str(val_loss) + 'model' + str(ii.item()))
            best_model = ii
            torch.save(model.state_dict(), f)
            print()

        # early stop - if validation loss does not increase for 15 epochs, stop learning process
        if opt.early_stop:
            if ii > opt.patience:
                if valid_mean_losses[ii - opt.patience] == np.min(valid_mean_losses[ii - opt.patience:]):
                    epochs_run = ii
                    break

        # Make predictions based on best model (lowest validation loss)
        # Load model
        # model.load_state_dict(torch.load(f))
        model.eval()

        # Threshold optimization on validation set
        thr = threshold_optimization(model, dev_dataloader_thr, gpu_id=opt.gpu_id)

        # Results on test set:
        matrix = evaluate(model, test_dataloader, thr, gpu_id=opt.gpu_id)

        # compute sensitivity and specificity for each class:
        MI_sensi = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
        MI_spec = matrix[0, 3] / (matrix[0, 3] + matrix[0, 2])
        STTC_sensi = matrix[1, 0] / (matrix[1, 0] + matrix[1, 1])
        STTC_spec = matrix[1, 3] / (matrix[1, 3] + matrix[1, 2])
        CD_sensi = matrix[2, 0] / (matrix[2, 0] + matrix[2, 1])
        CD_spec = matrix[2, 3] / (matrix[2, 3] + matrix[2, 2])
        HYP_sensi = matrix[3, 0] / (matrix[3, 0] + matrix[3, 1])
        HYP_spec = matrix[3, 3] / (matrix[3, 3] + matrix[3, 2])

        # compute mean sensitivity and specificity:
        mean_sensi = np.mean(matrix[:, 0]) / (np.mean(matrix[:, 0]) + np.mean(matrix[:, 1]))
        mean_spec = np.mean(matrix[:, 3]) / (np.mean(matrix[:, 3]) + np.mean(matrix[:, 2]))

        # print results:
        print('Final Test Results: \n ' + str(matrix) + '\n' + 'MI: sensitivity - ' + str(MI_sensi) + '; specificity - '
              + str(MI_spec) + '\n' + 'STTC: sensitivity - ' + str(STTC_sensi) + '; specificity - ' + str(STTC_spec)
              + '\n' + 'CD: sensitivity - ' + str(CD_sensi) + '; specificity - ' + str(CD_spec)
              + '\n' + 'HYP: sensitivity - ' + str(HYP_sensi) + '; specificity - ' + str(HYP_spec)
              + '\n' + 'mean: sensitivity - ' + str(mean_sensi) + '; specificity - ' + str(mean_spec))
        averages = (mean_sensi + mean_spec) / 2
        print('==================================================================================Mean: ', averages)
        if averages > max_acc:
            max_acc = averages
        print('==================================================================================Max acc is : ', max_acc)
        dt = datetime.now()
        with open('results/' + 'model' + str(best_model.item()) + '_' + str(datetime.timestamp(dt)) + '.txt', 'w') as f:
            print('Final Test Results: \n ' + str(matrix) + '\n' + 'MI: sensitivity - ' + str(MI_sensi) + '; specificity - '
                  + str(MI_spec) + '\n' + 'STTC: sensitivity - ' + str(STTC_sensi) + '; specificity - ' + str(STTC_spec)
                  + '\n' + 'CD: sensitivity - ' + str(CD_sensi) + '; specificity - ' + str(CD_spec)
                  + '\n' + 'HYP: sensitivity - ' + str(HYP_sensi) + '; specificity - ' + str(HYP_spec)
                  + '\n' + 'mean: sensitivity - ' + str(mean_sensi) + '; specificity - ' + str(mean_spec), file=f)

    # plot
    epochs_axis = torch.arange(1, epochs_run + 1)
    plot_losses(valid_mean_losses, train_mean_losses, ylabel='Loss',
                name='training-validation-loss-{}-{}-{}'.format(opt.learning_rate, opt.optimizer, dt))


if __name__ == '__main__':
    main()
