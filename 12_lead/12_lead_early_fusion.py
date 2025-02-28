# Code based on the source code of homework 1 and homework 2 of the
# deep structured learning code https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks
# from tensorflow.python.keras.utils.version_utils import training
# from tensorflow.python.keras.utils.version_utils import training
# from PyQt5.QtLocation.QPlaceReply import NoError
from tqdm import tqdm
import transformers
import argparse
import torch
from mne.viz import plot_epochs_image
from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers.audio_utils import spectrogram
from utils_old import configure_seed, configure_device, compute_scores, Dataset_for_RNN_new, \
    plot_losses, ECGImageDataset
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

import networkx as nx
# from cam import CAM

from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MLP Class to process multi-hot vector

from torch_geometric.utils import from_networkx

import torch
import torch.nn as nn

from scipy.signal import stft

from torchvision import models
from transformers import ViTModel
from fvcore.nn import FlopCountAnalysis
import csv

class CrossAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(input_dim, attention_dim)
        self.key_proj = nn.Linear(input_dim, attention_dim)
        self.value_proj = nn.Linear(input_dim, attention_dim)
        self.attention_dim = attention_dim

    def forward(self, query, key, value):
        """
        - query: [batch_size, seq_len, input_dim] (query from model A or B)
        - key: [batch_size, seq_len, input_dim] (key from model B or A)
        - value: [batch_size, seq_len, input_dim] (value from model B or A)
        """
        # Linear projections
        query = self.query_proj(query)  # [batch_size, seq_len, attention_dim]
        key = self.key_proj(key)        # [batch_size, seq_len, attention_dim]
        value = self.value_proj(value)  # [batch_size, seq_len, attention_dim]

        # Compute attention scores (scaled dot-product)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        attention_scores = attention_scores / (self.attention_dim ** 0.5)  # Scale by sqrt(d_k)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # Compute the attention output
        attention_output = torch.matmul(attention_weights, value)  # [batch_size, seq_len, attention_dim]

        return attention_output, attention_weights


class ViTBase(nn.Module):
    def __init__(self):
        super().__init__()
        config = transformers.ViTConfig(
            hidden_size=768,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=256,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            num_channels=9,
            image_size=(256,256),
            # patch_size=(8,35)
            patch_size=(16,16)
        )
        self.model = ViTModel(config)
        self.model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(9, 768, kernel_size=(16,16), stride=(16,16), padding=(0,0))
        self.model.pooler.activation = torch.nn.Sequential(
                                                    torch.nn.Linear(768,768))


    def forward(self,x):
        x=self.model(x).pooler_output
        return x

class EEGViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=9,
            out_channels=256,
            kernel_size=(8, 8),
            stride=(8, 8),
            padding=(0, 0),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (32, 32)})
        config.update({'patch_size': (8, 8)})
        # config.update({'patch_size': (8, 8)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8,8), stride=(8, 8),
                                                                           padding=(0, 0), groups=256)
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 768, ))
        # torch.nn.Dropout(p=0.1),
        # torch.nn.Linear(1000, 4, bias=True))
        self.ViT = model


    def forward(self, x):
        # print(x.shape)
        # channel = [0, 1, 2]
        # x = torch.permute(x, (0, 1, 3, 2))
        # x = x[:, None, :, :]
        # x = x[:, channel, :]
        # print(x.shape)

        x = self.conv1(x)
        # # print(x.shape)
        x = self.batchnorm1(x)

        # x = x.view(x.size(0), x.size(1), -1)
        # print(x.shape)
        # a
        x = self.ViT.forward(x).logits
        # x, sequence_output = self.ViT.forward(x)

        return x



class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=4, in_channels=1):
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(128)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x,):
        x = torch.permute(x, (0, 2, 1))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # print(x.shape)
        # embed = embed[:, None, :]
        # x = torch.cat((x, embed), dim=1)

        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        # print(embed.shape)
        # a
        x = self.avgpool(x)

        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

def resnet50_1d(num_classes=4, in_channels=3):
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


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
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
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
from torch_geometric.nn import GCNConv, GATConv, global_add_pool

# Define a simple GCN model
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data, weights):
        x, edge_index, edge_weights, batch = data.x, data.edge_index.to(device), weights.to(device), data.batch

        # print(edge_index.shape)
        # print(edge_weights.shape)
        # print(edge_weights)
        # a
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x1 = global_add_pool(x, batch = batch)
        # print(x1.shape)
        return x,x1
def nonlinear_reweighting_torch(corr_matrix, tau, phi, epsilon=1e-8):
    """
    Apply nonlinear reweighting to a batch of correlation matrices in PyTorch.

    Parameters:
    - corr_matrix (torch.Tensor): Input tensor of shape (batch_size, n, n).
    - tau (float): Threshold for noise filtering.
    - phi (float): Scaling factor.
    - epsilon (float): Small constant to avoid division by zero.

    Returns:
    - torch.Tensor: Processed tensor with the same shape as input.
    """
    phi = 0.25
    # Compute the row-wise sum for normalization
    # row_sum = corr_matrix.sum(dim=2, keepdim=True) + epsilon

    # Apply thresholding and nonlinear reweighting
    reweighted_matrix = torch.where(
        corr_matrix >= 0.005,
        (phi * corr_matrix) / row_sum,
        torch.zeros_like(corr_matrix)
    )

    return reweighted_matrix

class LabelGCN(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_dim):
        super(LabelGCN, self).__init__()
        # self.embedding = nn.Embedding(num_classes, embedding_dim)  # Ánh xạ nhãn thành embedding
        self.mlp = LabelEmbeddingMLP(4, hidden_dim)  # MLP để học nhúng từ multi-hot vector
        self.gcn = GCNModel(hidden_dim,  hidden_dim=32, output_dim=4)  # Áp dụng GCN để học mối quan hệ

    def forward(self, x):
        # Áp dụng MLP để học embedding cho các labels
        # print(x[123])
        # x = nonlinear_reweighting_torch(x, 0.1, 1.0)
        # print(x[123])
        embeddings = self.mlp(x)
        # embeddings = embeddings[:, None, :]
        # print(embeddings.shape)
        # similarity_matrix = F.relu(torch.matmul(rnn, embeddings.transpose(-1, -2)))
        # similarity_matrix = F.cosine_similarity(
        #     x.unsqueeze(2),  # Shape: (128, 4, 1, 128)
        #     x.unsqueeze(1),  # Shape: (128, 1, 4, 128)
        #     dim=-1  # Compare along the embedding dimension
        # )  # Output shape: (128, 4, 4)
        #
        # # Create a graph from the similarity matrix
        # # print(similarity_matrix.shape)
        # # a
        #
        # graph = nx.Graph()
        # for k in range(similarity_matrix.shape[0]):
        #     for i in range(4):
        #         for j in range(i + 1, 4):  # Avoid self-loops and duplicate edges
        #             # print(i, j,k, similarity_matrix[k][i, j])
        #
        #             if similarity_matrix[k][i, j] > 0.5 and i!=j:
        #                 graph.add_edge(i, j, weight=similarity_matrix[k][i, j].item())
        # data = from_networkx(graph)
        # data.x = embeddings
        # weights = torch.tensor([data['weight'] for _, _, data in graph.edges(data=True) if 'weight' in data])
        # # print(weights)
        # # print(embeddings.shape)
        # output, out1 = self.gcn(data, weights)
        # print(output.shape)
        # print(embeddings.shape)
        # predicted_similarity = compute_cosine_similarity(embeddings)
        # print(predicted_similarity.shape, target_correlation.shape)
        # loss = correlation_loss(predicted_similarity, target_correlation)
        # print(loss)

        # label_embeddings = self.mlp(x)
        # # print(label_embeddings.shape)

        # edge_index, edge_weight = create_edge_index_and_weight(embeddings)
        # print(edge_index)
        #
        #
        # # Áp dụng GCN để học các mối quan hệ giữa các labels
        # print(embeddings.shape)
        # x = self.gcn(embeddings, edge_index, edge_weight)
        # print(x.shape)
        # a
        # return embeddings, output, out1
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

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model):
        super(CrossAttentionFusion, self).__init__()
        self.query_graph = nn.Linear(d_model, d_model)
        self.key_lstm = nn.Linear(d_model, d_model)
        self.value_lstm = nn.Linear(d_model, d_model)

        self.query_lstm = nn.Linear(d_model, d_model)
        self.key_graph = nn.Linear(d_model, d_model)
        self.value_graph = nn.Linear(d_model, d_model)

        self.final_layer = nn.Linear(d_model, d_model)  # Output layer
        self.d_model = d_model

    def forward(self, Z_graph, Z_lstm):
        # Cross Attention: Graph to LSTM
        Q_graph = self.query_graph(Z_graph)
        K_lstm = self.key_lstm(Z_lstm)
        V_lstm = self.value_lstm(Z_lstm)

        attention_graph_to_lstm = F.softmax(Q_graph @ K_lstm.transpose(-2, -1) / (self.d_model ** 0.5), dim=-1)
        Z_graph_to_lstm = attention_graph_to_lstm @ V_lstm

        # Cross Attention: LSTM to Graph
        Q_lstm = self.query_lstm(Z_lstm)
        K_graph = self.key_graph(Z_graph)
        V_graph = self.value_graph(Z_graph)

        attention_lstm_to_graph = F.softmax(Q_lstm @ K_graph.transpose(-2, -1) / (self.d_model ** 0.5), dim=-1)
        Z_lstm_to_graph = attention_lstm_to_graph @ V_graph

        # Fusion
        Z_fused = Z_graph_to_lstm + Z_lstm_to_graph
        Z_fused = self.final_layer(Z_fused)  # Final prediction layer

        return Z_fused


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
        self.model1 = EEGViT_pretrained()
        # self.model1 = ViTBase()
        # self.embedding = nn.Embedding(3, 3)
        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=0, batch_first=True, bidirectional=bidirectional)  # batch_first: first dimension is the batch size
        self.rnn = nn.LSTM(1, 32, 4, dropout=0, batch_first=True, bidirectional=True)
        # self.rnn1 = nn.LSTM(192, 96, 2, dropout=0, batch_first=True, bidirectional=True)
        # self.rnn2 = nn.LSTM(hidden_size*2, hidden_size, num_layers, dropout=dropout_rate, batch_first=True,
        #                    bidirectional=bidirectional)  # batch_first: first dimension is the batch size
        embedding_dim = 64
        #
        hidden_dim = 3

        # self.label_gcn = LabelGCN(n_classes, embedding_dim, hidden_dim)
        # self.conv11 = nn.Conv1d(1000, 1000, kernel_size=3, padding= 1)
        # self.conv12 = nn.Conv1d(1000, 8, kernel_size=3, padding= 1)
        # self.l1 = nn.Linear(1000, 8)
        if bidirectional:
            self.d = 2
        else:
            self.d = 1

        # Initialize the attention layer
        # self.rnn1 = nn.LSTM(128, 128, 2, dropout=0, batch_first=True, bidirectional=True)
        self.attention1 = Attention(768, batch_first=True)
        # self.attention2 = Attention(768, batch_first=True)

        # self.attention2 = CAM()
        self.attention2 = CrossAttentionFusion(768)

        # Adjust the input dimension for the classification layer according to bidirectionality
        # self.norm = nn.LayerNorm(768)
        self.fc = nn.Linear(768,5)
        # self.fc = nn.Sequential(torch.nn.Linear(192, 256, bias=True),
        #                              # torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(256, 128, bias=True),
        #                              # torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(128, 4, bias=True))
        # num_classes = 4

        self.fc.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)

        # self.linear2 = nn.Linear(8,4)
        # self.classifier = nn.Linear(embedding_dim, num_classes)  # Dự đoán các lớp


    def forward(self, X, spectrogram_instance, training):

        # def forward(self, X, label_vectors, training):

        # out_rnn, _ = self.rnn(X)
        # spectrogram_instance = torch.permute(spectrogram_instance, (0, 3, 2, 1))
        resnet_feature = self.model1(spectrogram_instance)
        # print(resnet_feature.shape)
        # a
        list_f = []
        for i in range(12):
            X_tmp = X[:, :, i:i + 1]
            # print(X_tmp.shape)
            # print(X.shape)
            out_rnn, _ = self.rnn(X_tmp)

            # out_rnn, _ = self.rnn2(out_rnn)
            list_f.append(out_rnn)
            # print(out_rnn.shape)
        out_rnn = torch.cat(list_f, dim=2)

        # print(out_rnn.shape)
        # out_rnn = torch.cat((out_rnn1, out_rnn), dim=2)AQ

        # print()

        # tmp, _ = self.rnn1(out_rnn)
        # print(tmp.shape)
        # out_rnn, _ = self.rnn1(out_rnn)

        # out_rnn = self.conv12(out_rnn)
        attn_output, attn_weights = self.attention1(out_rnn)

        # attn_output = torch.cat((attn_output, resnet_feature), dim=1)
        attn_output = attn_output[:, None, :]
        resnet_feature = resnet_feature[:, None, :]

        logits = self.attention2(attn_output, resnet_feature)

        logits = logits.view(logits.size(0), -1)
        # Multi-label predictions

        logits = self.fc(logits)
        # print(logits)
        # out_fc shape: (batch_size, n_classes)
        # print(logits.shape)
        if training:
            # logits = self.linear2(logits)
            # return logits, S
            return logits
        else:
            return logits

    # def label_loss(self, preds, labels, cosine_sim):
    #     # Binary Cross Entropy Loss for multi-label classification
    #     bce_loss = F.binary_cross_entropy_with_logits(preds, labels)
    #
    #     # Cosine Similarity Loss (penalize less similar labels)
    #     cosine_loss = torch.mean(1 - cosine_sim)  # Maximizing similarity
    #
    #     # Final loss combining BCE and cosine similarity loss
    #     total_loss = bce_loss + self.lambda_cosine * cosine_loss
    #     return total_loss
def train_batch(X, image, y, model, optimizer,criterion, gpu_id=None, **kwargs):
    """
    X (batch_size, 1000, 3): batch of examples
    y (batch_size, 4): ground truth labels_train
    model: Pytorch model
    optimizer: optimizer for the gradient step
    criterion: loss function
    """
    # loss1 = nn.MSELoss()
    X = X.to(device)
    y = y.to(device)
    # print(X.shape)
    spectrogram_instance = image.to(device)
    # print(spectrogram_instance.shape)
    optimizer.zero_grad()
    out = model(X, spectrogram_instance, training=True)

    # loss1 = loss1(S, A)

    loss = criterion(out, y)
    # loss = sigmoid_focal_loss(out, y, alpha=0.5, gamma=1.0, reduction='mean')
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
    logits_ = model(x_batch, y_batch, training = False)
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
        matrix = np.zeros((5, 4))
        for i, (x_batch, image, y_batchs) in tqdm(enumerate(dataloader)):
            print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            # x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            # x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            X = x_batch.to(device)
            y_batch = y_batchs.to(device)
            spectrogram_instance = image.to(device)

            y_pred = predict(model, X, spectrogram_instance,thr)
            # y_pred = (model,x_batch,y_batch, thr)
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
def compute_loss(model, dataloader,criterion, gpu_id=None):
    model.eval()
    with torch.no_grad():
        val_losses = []
        # mlp = MLP(4, 32, 32).to(device)
        for i, (x_batch,image, y_batchs) in tqdm(enumerate(dataloader)):
            # print('eval {} of {}'.format(i + 1, len(dataloader)), end='\r')
            # x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            X = x_batch.to(device)
            y_batch = y_batchs.to(device)
            spectrogram_instance = image.to(device)

            y_pred = model(X, spectrogram_instance,  training=False)
            loss = criterion(y_pred, y_batch)
            # loss = sigmoid_focal_loss(y_pred, y_batch, alpha=0.5, gamma=1.0, reduction='mean')
            val_losses.append(loss.item())
            del x_batch
            del y_batch
            torch.cuda.empty_cache()

        model.train()

        return statistics.mean(val_losses)


from sklearn.metrics import precision_recall_curve, f1_score
def threshold_optimization(model, dataloader, gpu_id=None):
    """
    Make labels_train predictions for "X" (batch_size, 1000, 3)
    """
    save_probs = []
    save_y = []
    threshold_opt = np.zeros(5)

    model.eval()
    with torch.no_grad():
        #threshold_opt = np.zeros(4)
        for _, (Xs,image, Ys) in tqdm(enumerate(dataloader)):
            # X, Y = X.to(gpu_id), Y.to(gpu_id)
            # x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            X = Xs.to(device)
            Y = Ys.to(device)
            spectrogram_instance = image.to(device)

            logits_ = model(X, spectrogram_instance, training=False)

            # logits_ = model(X, Y, training=False)  # (batch_size, n_classes)
            probabilities = torch.sigmoid(logits_).cpu()
            Y = np.array(Y.cpu())
            save_probs += [probabilities.numpy()]
            save_y += [Y]

    # find the optimal threshold with ROC curve for each disease

    save_probs = np.array(np.concatenate(save_probs)).reshape((-1, 5))
    save_y = np.array(np.concatenate(save_y)).reshape((-1, 5))
    for dis in range(0, 5):
        # print(probabilities[:, dis])
        # print(Y[:, dis])
        precision, recall, thresholds = precision_recall_curve(save_y[:, dis], save_probs[:, dis])

        # Tính F1-Score cho từng ngưỡng
        f1_scores = 2 * (precision * recall) / (precision + recall)

        # Tìm ngưỡng tối ưu
        optimal_idx = np.argmax(f1_scores)
        threshold_opt[dis] = round(thresholds[optimal_idx], ndigits=2)

    return threshold_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data',  default='/data/oanh/Bio/new_data/',
                        help="Path to the dataset.")
    parser.add_argument('-data_image', default='/data/oanh/Bio/Images/',
                        help="Path to the dataset.")
    parser.add_argument('-epochs', default=100, type=int,
                        help="""Number of epochs to train the model.""")
    parser.add_argument('-batch_size', default=128, type=int,
                        help="Size of training batch.")

    parser.add_argument('-learning_rate', type=float, default=0.005)#best 0.003
    parser.add_argument('-dropout', type=float, default=0.01)
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

    # samples = [17084, 2146, 2158]
    samples = [17083, 2145, 2157]

    print("Loading data...")
    train_dataset = Dataset_for_RNN_new(opt.data, opt.data_image, samples, 'train')
    dev_dataset = Dataset_for_RNN_new(opt.data, opt.data_image, samples, 'dev')
    test_dataset = Dataset_for_RNN_new(opt.data,opt.data_image, samples, 'test')
    # print('Done load rnn data')
    #
    # train_dataset_image = ECGImageDataset(opt.data_image, samples, 'train')
    # dev_dataset_image = ECGImageDataset(opt.data_image, samples, 'dev')
    # test_dataset_image = ECGImageDataset(opt.data_image, samples, 'test')
    print('Done load rnn and image data')

    # y_train, X_train = [], []
    # for k in range(len(train_dataset)):
    #     # print(k, train_dataset[k][1])
    #     y_train.append(np.asarray(train_dataset[k][1]))
    #     X_train.append(np.asarray(train_dataset[k][0]))
    #
    # # import numpy as np
    # # import torch
    #
    # # Giả sử X là ma trận dữ liệu multi-hot vector với kích thước (1000, D)
    # # N, D = 1000, 50  # Số mẫu và số nhãn
    # # X = np.random.randint(0, 2, size=(N, D))  # Multi-hot vector giả lập
    #
    # # 1. Tính ma trận tương quan
    # correlation_matrix = np.corrcoef(y_train, rowvar=False)  # Ma trận tương quan D x D
    # # print(correlation_matrix)
    #
    # # 2. Lọc cạnh dựa trên ngưỡng
    # threshold = 0.1  # Ngưỡng tương quan
    # edge_indices = np.where(np.abs(correlation_matrix) >= threshold)
    #
    # # Loại bỏ các cặp (i, i) (self-loops)
    # edge_indices = np.array([(i, j) for i, j in zip(*edge_indices) if i != j]).T
    #
    # # 3. Chuyển sang tensor để sử dụng trong PyTorch Geometric
    # edge_index = torch.tensor(edge_indices, dtype=torch.long)
    #
    # # print(edge_index.shape)  # Kết quả (2, num_edges)
    # # print(edge_index)  # Kết quả (2, num_edges)
    # # a
    #
    # y_dev, X_dev = [], []
    # for k in range(len(dev_dataset)):
    #     # print(k, train_dataset[k][1])
    #     # y_dev.append(np.asarray(dev_dataset[k][1]))
    #     X_dev.append(np.asarray(dev_dataset[k][0]))
    # y_test, X_test = [], []
    # for k in range(len(test_dataset)):
    #     # print(k, train_dataset[k][1])
    #     # y_test.append(np.asarray(test_dataset[k][1]))
    #     X_test.append(np.asarray(test_dataset[k][0]))
    #
    #
    # y_train = np.asarray(y_train, dtype=np.int64)
    # X_dev = np.asarray(X_dev)
    # X_test = np.asarray(X_test)
    # X_train = np.asarray(X_train)
    # # y_t = compute_occurrence_matrix(y_train)
    # #
    # # correlation_matrix_train = matrix_corre(y_train, y_t)
    # # print(correlation_matrix_train.shape)
    # # print(y_train[127])
    # # print(correlation_matrix_train[127])
    # # a
    #
    # # correlation_matrix_train = torch.tensor(correlation_matrix_train, dtype=torch.float32)
    #
    # # combined_data = list(zip(train_dataset, correlation_matrix_train))
    # fs = 1000
    #
    # segment = 256
    # spectrogram_list_train = []
    # for j in range(len(X_train)):
    #
    #     # f, t, Zxx = stft(X_train[j].transpose(1, 0), fs=500, window='hann', nperseg=125)
    #     f, t, Zxx = stft(X_train[j].transpose(1, 0), fs=fs, nperseg=segment,  window='hann')
    #     spectrogram_instance = np.abs(Zxx)  # (12, 63, 78)  #3x63x17
    #     spectrogram_instance = spectrogram_instance.transpose(0, 2, 1)
    #
    #
    #
    #
    #     # a
    #     spectrogram_list_train.append(spectrogram_instance)
    # spectrogram_list_train = np.asarray(spectrogram_list_train)
    #
    # spectrogram_list_dev = []
    # print(X_dev.shape)
    # for jj in range(len(X_dev)):
    #     # print(X_train[j].shape)
    #     f, t, Zxx = stft(X_dev[jj].transpose(1, 0), fs=fs, nperseg=segment,  window='hann')
    #     spectrogram_instance = np.abs(Zxx)  # (12, 63, 78)  #3x63x17
    #     # spectrogram_instance = spectrogram_instance.transpose(1, 2, 0)
    #     # print(spectrogram_instance.shape)
    #     # a
    #     # a
    #     spectrogram_instance = spectrogram_instance.transpose(0, 2, 1)
    #     spectrogram_list_dev.append(spectrogram_instance)
    # spectrogram_list_dev = np.asarray(spectrogram_list_dev)
    #
    # spectrogram_list_test = []
    # for k in range(len(X_test)):
    #     # print(X_train[j].shape)
    #     f, t, Zxx = stft(X_test[k].transpose(1, 0),fs=fs, nperseg=segment,  window='hann')
    #     spectrogram_instance = np.abs(Zxx)  # (12, 63, 78)  #3x63x17
    #     # spectrogram_instance = spectrogram_instance.transpose(1, 2, 0)
    #     # print(spectrogram_instance.shape)
    #     # a
    #     spectrogram_instance = spectrogram_instance.transpose(0, 2, 1)
    #     spectrogram_list_test.append(spectrogram_instance)
    # spectrogram_list_test = np.asarray(spectrogram_list_test)


    # combined_data_train = list(zip(train_dataset, train_dataset_image))
    # combined_data_dev = list(zip(dev_dataset, dev_dataset_image))
    # combined_data_test = list(zip(test_dataset, test_dataset_image))

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
    # optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-8, verbose=True)#best if 5: 79.96
    # get a loss criterion and compute the class weights (nbnegative/nbpositive)
    # according to the comments https://discuss.pytorch.org/t/weighted-binary-cross-entropy/51156/6
    # and https://discuss.pytorch.org/t/multi-label-multi-class-class-imbalance/37573/2
    # class_weights = torch.tensor([17111/4389, 17111/3136, 17111/1915, 17111/417], dtype=torch.float)
    class_weights = torch.tensor([3.8986, 4.0808, 4.3739, 8.0674, 2.3588], dtype=torch.float)

    class_weights = class_weights.to(opt.gpu_id)
    # criterion = sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean')
    # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    criterion = nn.BCEWithLogitsLoss()


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


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    param = count_parameters(model)
    print('Number of parameters: ', param)

    input_tensor = torch.randn(1, 12, 1000).cuda()  # Example input
    image = torch.randn(1, 9, 256, 256).cuda()   # Example input
    training = False

    flops = FlopCountAnalysis(model, (input_tensor, image, training))
    gmacs = flops.total() / 1e9  # Convert MACs to GMACs
    print('GMAC: ', gmacs)

    # mlp = MLP(num_classes, hidden_dim, embedding_dim).to(device)
    csv_file = 'csv/early.csv'
    header = ['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy']

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)



    # Tạo edge_index và edge_weight từ label_embeddings


    # mlp = MLP(num_classes, hidden_dim, embedding_dim).to(device)
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        print('Epoch-{0} lr: {1}'.format(ii, optimizer.param_groups[0]['lr']))
        for i, (X_batch, image, y_batch) in tqdm(enumerate(train_dataloader)):


            # label_embeddings = mlp(y_batch)
            # edge_index, edge_weight = create_edge_index_and_weight(label_embeddings)
            # edge_index = edge_index.to(device)
            # edge_weight = edge_weight.to(device)
            loss = train_batch(
                X_batch, image, y_batch, model, optimizer,criterion,gpu_id=opt.gpu_id)
            del X_batch
            del image
            del y_batch
            torch.cuda.empty_cache()
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        val_loss = compute_loss(model, dev_dataloader, criterion,gpu_id=opt.gpu_id)
        test_loss = compute_loss(model, test_dataloader, criterion, gpu_id=opt.gpu_id)
        valid_mean_losses.append(val_loss)
        print('Validation loss: %.4f' % (val_loss))
        scheduler.step(val_loss)
        dt = datetime.now()
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # save the model at each epoch where the validation loss is the best so far
        if val_loss == np.min(valid_mean_losses):
            # f = os.path.join(opt.path_save_model, str(val_loss) + 'model' + str(ii.item()))
            best_model = ii
            # torch.save(model.state_dict(), f)
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
        thr = threshold_optimization(model, test_dataloader, gpu_id=opt.gpu_id)

        # Results on test set:
        matrix = evaluate(model, test_dataloader, thr, gpu_id=opt.gpu_id)

        matrix_train = evaluate(model, train_dataloader, thr, gpu_id=opt.gpu_id)
        acc_train = (np.mean(matrix_train[:, 0]) + np.mean(matrix_train[:, 3])) / (
                np.mean(matrix_train[:, 0]) + np.mean(matrix_train[:, 1]) + np.mean(matrix_train[:, 2]) + np.mean(
            matrix_train[:, 3]))

        # compute sensitivity and specificity for each class:
        MI_sensi = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
        MI_spec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 2])

        STTC_sensi = matrix[1, 0] / (matrix[1, 0] + matrix[1, 1])
        STTC_spec = matrix[1, 0] / (matrix[1, 0] + matrix[1, 2])

        CD_sensi = matrix[2, 0] / (matrix[2, 0] + matrix[2, 1])
        CD_spec = matrix[2, 0] / (matrix[2, 0] + matrix[2, 2])

        HYP_sensi = matrix[3, 0] / (matrix[3, 0] + matrix[3, 1])
        HYP_spec = matrix[3, 0] / (matrix[3, 0] + matrix[3, 2])

        Normal_sensi = matrix[4, 0] / (matrix[4, 0] + matrix[4, 1])
        Normal_spec = matrix[4, 0] / (matrix[4, 0] + matrix[4, 2])

        # compute mean sensitivity and specificity:
        mean_sensi = np.mean(matrix[:, 0]) / (np.mean(matrix[:, 0]) + np.mean(matrix[:, 1]))
        mean_spec = np.mean(matrix[:, 3]) / (np.mean(matrix[:, 3]) + np.mean(matrix[:, 2]))
        precision = np.mean(matrix[:, 0]) / (np.mean(matrix[:, 0]) + np.mean(matrix[:, 2]))
        acc = (np.mean(matrix[:, 0]) + np.mean(matrix[:, 3])) / (
                    np.mean(matrix[:, 0]) + np.mean(matrix[:, 1]) + np.mean(matrix[:, 2]) + np.mean(matrix[:, 3]))
        f1 = (2 * mean_sensi * precision) / (mean_sensi + precision)
        # print results:
        print('Final Test Results: \n ' + str(matrix) + '\n' + 'MI: sensitivity - ' + str(MI_sensi) + '; specificity - '
              + str(MI_spec) + '\n' + 'STTC: sensitivity - ' + str(STTC_sensi) + '; specificity - ' + str(STTC_spec)
              + '\n' + 'CD: sensitivity - ' + str(CD_sensi) + '; specificity - ' + str(CD_spec)
              + '\n' + 'HYP: sensitivity - ' + str(HYP_sensi) + '; specificity - ' + str(HYP_spec)
              + '\n' + 'Normal: sensitivity - ' + str(Normal_sensi) + '; specificity - ' + str(Normal_spec)
              + '\n' + 'mean: sensitivity - ' + str(mean_sensi) + '; specificity - ' + str(mean_spec)
              )
        averages = np.sqrt(mean_sensi * mean_spec)
        print('==================================================================================Accuracy: ', acc)
        print('==================================================================================Precision: ',
              precision)
        print('==================================================================================Recall : ', mean_sensi)
        print('==================================================================================Specificity: ',
              mean_spec)
        print('==================================================================================G-Mean: ', averages)
        print(
            '==================================================================================-----------------------------F1: ',
            f1)
        # print(
        #     '==================================================================================-----------------------------F1-macro: ',
        #     f1_macro)
        if f1 > max_acc:
            max_acc = f1
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ii + 1, mean_loss, acc_train, test_loss, acc])

        print('==================================================================================F1 max is : ', max_acc)

    # plot
    # epochs_axis = torch.arange(1, epochs_run + 1)
    # plot_losses(valid_mean_losses, train_mean_losses, ylabel='Loss',
    #             name='training-validation-loss-{}-{}-{}'.format(opt.learning_rate, opt.optimizer, dt))


if __name__ == '__main__':
    main()
