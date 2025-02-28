# code based on the source code of homework 1 and homework 2 of the
# deep structured learning code https://fenix.tecnico.ulisboa.pt/disciplinas/AEProf/2021-2022/1-semestre/homeworks

# import the necessary packages
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import csv
import tifffile
from PIL import Image


def configure_device(gpu_id):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def plot(plottable, ylabel="", name=""):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(plottable)
    plt.savefig("%s.pdf" % (name), bbox_inches="tight")


def plot_losses(valid_losses, train_losses, ylabel="", name=""):
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    # plt.xticks(epochs)
    plt.plot(valid_losses, label="validation")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.savefig("%s.pdf" % (name), bbox_inches="tight")


# create a generator to read the images as we train the model
# (similar to flow_from_directory Keras)
class ECGImageDataset(Dataset):
    """
    path/train/images
              /labels
        /val/images
            /labels
        /test/images
             /labels
    """

    def __init__(self, path, train_dev_test, part="train"):
        self.path = path
        self.part = part
        self.train_dev_test = train_dev_test

    def __len__(self):
        if self.part == "train":
            return self.train_dev_test[0]
        elif self.part == "dev":
            return self.train_dev_test[1]
        elif self.part == "test":
            return self.train_dev_test[2]

    def __getitem__(self, idx):
        X = read_data_for_CNN(self.path, self.part, idx)
        return torch.tensor(X).float()

from skimage import io
def read_data_for_CNN(path, partition, idx):
    """Read the ECG Image Data"""
    path_labels = str(path) + "labels_" + str(partition)
    path_X = str(path) + "X_cnn_" + str(partition)
    index = idx
    # label = np.load(str(path_labels) + "/" + str(index) + ".npy")
    # image = tifffile.imread(str(path_X) + "/" + str(index) + ".tif")

    image = io.imread(str(path_X) + "/" + str(index) + ".tif")
    image = np.array(image)

    image = image / 255.0  # normalization
    return image


class Dataset_for_RNN(Dataset):
    """
    path/labels_train
        /X_train
        /labels_val
        /X_val
        /labels_test
        /X_test
    """


    def __init__(self, path, train_dev_test, part="train"):
        self.path = path
        self.part = part
        self.train_dev_test = train_dev_test

    def __len__(self):
        if self.part == "train":
            return self.train_dev_test[0] - 1
        elif self.part == "dev":
            return self.train_dev_test[1]- 1
        elif self.part == "test":
            return self.train_dev_test[2]- 1

    def __getitem__(self, idx):
        X, y = read_data_for_RNN(self.path, self.part, idx)
        return torch.tensor(X).float(), torch.tensor(y).float()


def read_data_for_RNN(path, partition, idx):
    path_labels = str(path) + "labels_" + str(partition)
    path_X = str(path) + "X_rnn_" + str(partition)
    index = idx
    label = np.load(str(path_labels) + "/" + str(index) + ".npy")
    X = np.load(str(path_X) + "/" + str(index) + ".npy")
    return X, label


class Dataset_for_RNN_new(Dataset):

    def __init__(self, path, path_image, train_dev_test, part="train"):
        self.path_rnn = path
        self.path_image = path_image
        self.part = part
        self.train_dev_test = train_dev_test
        print(self.train_dev_test)


    def __len__(self):
        if self.part == "train":
            return self.train_dev_test[0]
        elif self.part == "dev":
            return self.train_dev_test[1]
        elif self.part == "test":
            return self.train_dev_test[2]

    def __getitem__(self, idx):
        # print(idx)
        X, image, y = read_data_for_RNN_CNN(self.path_rnn,self.path_image, self.part, idx)
        return torch.tensor(X).float(),torch.tensor(image).float(), torch.tensor(y).float()

def read_data_for_RNN_CNN(path_rnn, path_image, partition, idx):
    path_labels = str(path_rnn) + "labels_" + str(partition)
    path_X = str(path_rnn) + "X_rnn_" + str(partition)
    index = idx
    label = np.load(str(path_labels) + "/" + str(index) + ".npy")
    # X = np.load(str(path_X) + "/" + str(index) + ".npy")#normal
    X = np.load(str(path_X) + "/" + str(index) + ".npy")


    # new
    path_X = str(path_image) + "X_cnn_" + str(partition)
    # index = idx
    # label = np.load(str(path_labels) + "/" + str(index) + ".npy")
    # image = tifffile.imread(str(path_X) + "/" + str(index) + ".tif")

    image = io.imread(str(path_X) + "/" + str(index) + ".tif")
    image = np.array(image)

    image = image / 255.0  # normalization
    image = np.tile(image, (4, 1, 1))
    # print(image.shape)
    # a
    return X, image, label

def read_data_for_RNN(path, partition, idx):
    path_labels = str(path) + "labels_" + str(partition)
    path_X = str(path) + "X_rnn_" + str(partition)
    index = idx
    label = np.load(str(path_labels) + "/" + str(index) + ".npy")
    # X = np.load(str(path_X) + "/" + str(index) + ".npy")#normal
    X = np.load(str(path_X) + "/" + str(index) + ".npy")

    # print(X.shape)

    # X = np.fft.fft(X, axis=0)
    # X = np.abs(X)
    return X, label


# performance evaluation, compute the tp, fn, fp, and tp for each disease class
# and compute the specificity and sensitivity
# def compute_scores(y_true, y_pred, matrix):
#     # print(matrix.shape)
#     for j in range(len(y_true)):
#         pred = y_pred[j]
#         gt = y_true[j]
#         for i in range(0, 5):  # for each class
#             matrix = computetpfnfp(pred, gt, i, matrix, 4)
#             # matrix = computetpfnfp_ori(pred[i], gt[i], i, matrix)
#     return matrix


def compute_scores(y_true, y_pred, matrix):
    for j in range(len(y_true)):
        pred = y_pred[j]
        gt = y_true[j]
        for i in range(0, 5):  # for each class
            matrix = computetpfnfp(pred[i], gt[i], i, matrix)
    return matrix


def computetpfnfp(pred, gt, i, matrix):
    if gt == 0 and pred == 0:  # tn
        matrix[i, 3] += 1
    if gt == 1 and pred == 0:  # fn
        matrix[i, 1] += 1
    if gt == 0 and pred == 1:  # fp
        matrix[i, 2] += 1
    if gt == 1 and pred == 1:  # tp
        matrix[i, 0] += 1
    return matrix
def compute_scores_with_norm(y_true, y_pred, matrix, norm_vec):
    for j in range(len(y_true)):
        pred = y_pred[j]
        gt = y_true[j]
        norm_pred = True
        norm_gt = True
        for i in range(0, 5):  # for each class
            matrix = computetpfnfp(pred[i], gt[i], i, matrix)
            if gt[i] == 1 & norm_gt:
                norm_gt = False
            if pred[i] == 1 & norm_pred:
                norm_pred = False
        if norm_gt == 0 and norm_pred == 0:  # tn
            norm_vec[3] += 1
        if norm_gt == 1 and norm_pred == 0:  # fn
            norm_vec[1] += 1
        if norm_gt == 0 and norm_pred == 1:  # fp
            norm_vec[2] += 1
        if norm_gt == 1 and norm_pred == 1:  # tp
            norm_vec[0] += 1
    return matrix, norm_vec


def compute_scores_dev(matrix):
    matrix[matrix == 0] = 0.01
    # print(matrix)
    sensitivity = matrix[:, 0] / (matrix[:, 0] + matrix[:, 1])  # tp/(tp+fn)
    specificity = matrix[:, 3] / (matrix[:, 3] + matrix[:, 2])  # tn/(tn+fp)
    return np.mean(sensitivity), np.mean(specificity)


def computetpfnfp_ori(pred, gt, i, matrix):
    if gt == 0 and pred == 0:  # tn
        matrix[i, 3] += 1
    if gt == 1 and pred == 0:  # fn
        matrix[i, 1] += 1
    if gt == 0 and pred == 1:  # fp
        matrix[i, 2] += 1
    if gt == 1 and pred == 1:  # tp
        matrix[i, 0] += 1
    return matrix


def computetpfnfp_(pred, gt, i, matrix, num_classes):
    """
    Cập nhật ma trận tính TP, FN, FP, TN cho từng lớp trong bài toán multi-label.

    Parameters:
    - pred: Dự đoán (dạng vector).
    - gt: Ground truth (dạng vector).
    - i: Lớp đang xử lý.
    - matrix: Ma trận chứa giá trị TP, FN, FP, TN.
    - num_classes: Tổng số lớp.

    Returns:
    - matrix: Ma trận sau khi cập nhật.
    """
    # Xử lý cho từng lớp (class-specific)
    # print(i)
    if i != num_classes:
        if gt[i] == 0 and pred[i] == 0:  # tn
            matrix[i, 3] += 1
        if gt[i] == 1 and pred[i] == 0:  # fn
            matrix[i, 1] += 1
        if gt[i] == 0 and pred[i] == 1:  # fp
            matrix[i, 2] += 1
        if gt[i] == 1 and pred[i] == 1:  # tp
            matrix[i, 0] += 1
    # Xử lý riêng cho lớp "Norm" (toàn bộ nhãn đều là 0)
    if i == num_classes:  # Lớp "Norm" được xử lý cuối cùng
        is_norm_gt = all(label == 0 for label in gt)  # Ground truth là Norm
        is_norm_pred = all(label == 0 for label in pred)  # Prediction là Norm

        if is_norm_gt and is_norm_pred:  # tp
            matrix[i, 0] += 1
        if is_norm_gt and not is_norm_pred:  # fn
            matrix[i, 1] += 1
        if not is_norm_gt and is_norm_pred:  # fp
            matrix[i, 2] += 1
        if not is_norm_gt and not is_norm_pred:  # tn
            matrix[i, 3] += 1
        # print(gt)
        # print(pred)
        # print(matrix)


    return matrix


def compute_save_metrics(matrix, matrix_dev, opt_threshold, date, epoch, strategy, path_save_model, learning_rate,
                         optimizer, dropout, epochs, hidden_size, batch_size, test_id):

    # compute sensitivity and specificity for each class:
    MI_sensi = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    MI_spec = matrix[0, 3] / (matrix[0, 3] + matrix[0, 2])
    STTC_sensi = matrix[1, 0] / (matrix[1, 0] + matrix[1, 1])
    STTC_spec = matrix[1, 3] / (matrix[1, 3] + matrix[1, 2])
    CD_sensi = matrix[2, 0] / (matrix[2, 0] + matrix[2, 1])
    CD_spec = matrix[2, 3] / (matrix[2, 3] + matrix[2, 2])
    HYP_sensi = matrix[3, 0] / (matrix[3, 0] + matrix[3, 1])
    HYP_spec = matrix[3, 3] / (matrix[3, 3] + matrix[3, 2])

    MI_sensi_dev = matrix_dev[0, 0] / (matrix_dev[0, 0] + matrix_dev[0, 1])
    MI_spec_dev = matrix_dev[0, 3] / (matrix_dev[0, 3] + matrix_dev[0, 2])
    STTC_sensi_dev = matrix_dev[1, 0] / (matrix_dev[1, 0] + matrix_dev[1, 1])
    STTC_spec_dev = matrix_dev[1, 3] / (matrix_dev[1, 3] + matrix_dev[1, 2])
    CD_sensi_dev = matrix_dev[2, 0] / (matrix_dev[2, 0] + matrix_dev[2, 1])
    CD_spec_dev = matrix_dev[2, 3] / (matrix_dev[2, 3] + matrix_dev[2, 2])
    HYP_sensi_dev = matrix_dev[3, 0] / (matrix_dev[3, 0] + matrix_dev[3, 1])
    HYP_spec_dev = matrix_dev[3, 3] / (matrix_dev[3, 3] + matrix_dev[3, 2])

    # compute mean sensitivity and specificity:
    mean_sensi = np.mean(matrix[:, 0]) / (np.mean(matrix[:, 0]) + np.mean(matrix[:, 1]))
    mean_spec = np.mean(matrix[:, 3]) / (np.mean(matrix[:, 3]) + np.mean(matrix[:, 2]))
    mean_sensi_dev = np.mean(matrix_dev[:, 0]) / (np.mean(matrix_dev[:, 0]) + np.mean(matrix_dev[:, 1]))
    mean_spec_dev = np.mean(matrix_dev[:, 3]) / (np.mean(matrix_dev[:, 3]) + np.mean(matrix_dev[:, 2]))

    # print results:
    print(
        "Final Validation Results: \n "
        + str(matrix_dev)
        + "\n"
        + "MI: sensitivity - "
        + str(MI_sensi_dev)
        + "; specificity - "
        + str(MI_spec_dev)
        + "\n"
        + "STTC: sensitivity - "
        + str(STTC_sensi_dev)
        + "; specificity - "
        + str(STTC_spec_dev)
        + "\n"
        + "CD: sensitivity - "
        + str(CD_sensi_dev)
        + "; specificity - "
        + str(CD_spec_dev)
        + "\n"
        + "HYP: sensitivity - "
        + str(HYP_sensi_dev)
        + "; specificity - "
        + str(HYP_spec_dev)
        + "\n"
        + "mean: sensitivity - "
        + str(mean_sensi_dev)
        + "; specificity - "
        + str(mean_spec_dev)
    )

    print(
        "Final Test Results: \n "
        + str(matrix)
        + "\n"
        + "MI: sensitivity - "
        + str(MI_sensi)
        + "; specificity - "
        + str(MI_spec)
        + "\n"
        + "STTC: sensitivity - "
        + str(STTC_sensi)
        + "; specificity - "
        + str(STTC_spec)
        + "\n"
        + "CD: sensitivity - "
        + str(CD_sensi)
        + "; specificity - "
        + str(CD_spec)
        + "\n"
        + "HYP: sensitivity - "
        + str(HYP_sensi)
        + "; specificity - "
        + str(HYP_spec)
        + "\n"
        + "mean: sensitivity - "
        + str(mean_sensi)
        + "; specificity - "
        + str(mean_spec)
    )
    avg = (mean_sensi +  mean_spec)/2
    print('==========================================Mean: ', avg)

    with open(
        "{}{}{}_{}_ep{}_lr{}_opt{}_dr{}_eps{}_hs{}_bs{}.txt".format(
            path_save_model,
            test_id,
            strategy,
            date,
            epoch.item(),
            learning_rate,
            optimizer,
            dropout,
            epochs,
            hidden_size,
            batch_size,
        ),
        "w",
    ) as f:
        f.write("Final Results\n\n")
        f.write("Threshold: {}\n\n".format(np.round(opt_threshold, 4)))

        f.write("Development/Validation\n")
        f.write("MI\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(MI_sensi_dev, MI_spec_dev))
        f.write("STTC\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(STTC_sensi_dev, STTC_spec_dev))
        f.write("CD\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(CD_sensi_dev, CD_spec_dev))
        f.write("HYP\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(HYP_sensi_dev, HYP_spec_dev))
        f.write("Mean\n\tSensitivity: {}\n\tSpecificity: {}\n\n\n".format(mean_sensi_dev, mean_spec_dev))

        f.write("Test\n")
        f.write("MI\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(MI_sensi, MI_spec))
        f.write("STTC\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(STTC_sensi, STTC_spec))
        f.write("CD\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(CD_sensi, CD_spec))
        f.write("HYP\n\tSensitivity: {}\n\tSpecificity: {}\n\n".format(HYP_sensi, HYP_spec))
        f.write("Mean\n\tSensitivity: {}\n\tSpecificity: {}".format(mean_sensi, mean_spec))

    fields = [test_id,
              strategy,
              date,
              epoch.item(),
              learning_rate,
              optimizer,
              dropout,
              epochs,
              hidden_size,
              batch_size,
              mean_sensi_dev,
              mean_spec_dev,
              mean_sensi,
              mean_spec
              ]

    with open(path_save_model + "auto_results.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


import torch.nn.functional as F
import numpy as np

def calculate_metrics(test_loader, model, num1):
    DSC = 0.0
    IoU_sum = 0.0
    Recall_sum = 0.0
    Precision_sum = 0.0

    model.eval()  # Đặt mô hình ở chế độ đánh giá
    with torch.no_grad():  # Tắt tính toán gradient
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, res1 = model(image)
            # Upsample kết quả về kích thước ground truth
            res = F.interpolate(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
            res = torch.sigmoid(res).cpu().numpy().squeeze()

            # Nếu sử dụng soft Dice
            input = res
            target = gt
            smooth = 1.0

            # Tính Dice Coefficient
            intersection = (input * target).sum()
            dice = (2.0 * intersection + smooth) / (input.sum() + target.sum() + smooth)
            DSC += dice

            # Tính IoU
            union = input.sum() + target.sum() - intersection
            IoU = (intersection + smooth) / (union + smooth)
            IoU_sum += IoU

            # Tính Precision và Recall
            tp = intersection  # True Positives
            fp = input.sum() - tp  # False Positives
            fn = target.sum() - tp  # False Negatives

            precision = (tp + smooth) / (tp + fp + smooth)
            recall = (tp + smooth) / (tp + fn + smooth)
            Precision_sum += precision
            Recall_sum += recall

    # Tính trung bình các giá trị
    mean_dice = DSC / num1
    mean_IoU = IoU_sum / num1
    mean_precision = Precision_sum / num1
    mean_recall = Recall_sum / num1

    return mean_dice, mean_IoU, mean_precision, mean_recall