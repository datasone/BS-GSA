import matplotlib.pyplot as plt
import pandas
import pytorch_lightning as pl
import seaborn
import torch
import torch.nn.functional as F
import torchsummary
from torch import nn
from torchmetrics import functional as FM
from .CoordAtt import CoordAtt
from .FramePooling import FramePooling
import os


class CATP(pl.LightningModule):
    def __init__(self, target_dim, max_length, out_types=4, lr=1e-4, dropout_rate=0.5, t=2, aacnn_mode=False, frame_pooling_enabled=False):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr

        self.aacnn_mode = aacnn_mode

        self.frame_pooling_enabled = frame_pooling_enabled

        if not aacnn_mode:
            self.conv_layers = nn.Sequential(
                CoordAtt(1, 16),
                nn.Conv2d(16, 16, (5, 5), padding="same"),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                CoordAtt(16, 16),
                nn.Conv2d(16, 32, (3, 3), padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                CoordAtt(32, 32),
                nn.Conv2d(32, 64, (3, 3), padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                CoordAtt(64, 64),
                nn.Conv2d(64, 80, (3, 3), padding="same"),
                nn.BatchNorm2d(80),
                nn.Dropout(dropout_rate),
            )
        else:
            self.conv_1a = nn.Sequential(
                nn.Conv2d(1, 16, (10, 2), padding="same"),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )

            self.conv_1b = nn.Sequential(
                nn.Conv2d(1, 16, (2, 8), padding="same"),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )

            self.replace_ca = nn.Sequential(
                nn.Conv2d(1, 16, (3, 3), padding="same"),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                CoordAtt(16, 16),
            )

            self.conv_layers = nn.Sequential(
                nn.Conv2d(16, 32, (3, 3), padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                CoordAtt(32, 32),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(32, 48, (3, 3), padding="same"),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(48, 64, (3, 3), padding="same"),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 80, (3, 3), padding="same"),
                nn.BatchNorm2d(80),
                nn.ReLU(),
                CoordAtt(80, 80),
            )

            if self.frame_pooling_enabled:
                self.frame_pooling = nn.Sequential(
                    FramePooling(target_channel=max_length, linear_dim=target_dim),
                    nn.Dropout(dropout_rate),
                )

        if aacnn_mode:
            self.classification_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(128 / 4 * (t * 40) / 4 * 80 * 2), out_types),
            )
        else:
            self.classification_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(128 / 2 * (t * 40) / 2 * 80), out_types),
            )

    def forward(self, x):
        if self.aacnn_mode:
            conv_1a = self.conv_1a(x)
            conv_1b = self.conv_1b(x)
            conv_features = torch.cat([conv_1a, conv_1b], 2)
            # conv_features = self.replace_ca(x)
            conv_output = self.conv_layers(conv_features)
        else:
            conv_output = self.conv_layers(x)

        output = self.classification_layers(conv_output)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch  # [batch_size, frames, 1, 128, 81]
        if self.frame_pooling_enabled:
            y_hat = []
            for xi in x:
                conv_output = []
                for xii in xi:
                    # xii = torch.unsqueeze(xii, dim=0)
                    # conv_1a = self.conv_1a(xii)
                    # conv_1b = self.conv_1b(xii)
                    # conv_features = torch.cat([conv_1a, conv_1b], 2)
                    conv_features = self.replace_ca(x)
                    conv_output.append(self.conv_layers(conv_features))
                conv_output = torch.cat(conv_output, dim=1)
                model_output = self.frame_pooling(conv_output)
                y_hat.append(self.classification_layers(model_output))

            y_hat = torch.cat(y_hat, dim=0)
            y = torch.tensor(y)
        else:
            y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch  # [batch_size, frames, 1, 128, 81]
        if self.frame_pooling_enabled:
            y_hat = []
            for xi in x:
                conv_output = []
                for xii in xi:
                    xii = torch.unsqueeze(xii, dim=0)
                    conv_1a = self.conv_1a(xii)
                    conv_1b = self.conv_1b(xii)
                    conv_features = torch.cat([conv_1a, conv_1b], 2)
                    conv_output.append(self.conv_layers(conv_features))
                conv_output = torch.cat(conv_output, dim=1)
                model_output = self.frame_pooling(conv_output)
                y_hat.append(self.classification_layers(model_output))

            y_hat = torch.cat(y_hat, dim=0)
            y = torch.tensor(y)
        else:
            y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        self.log("valid_loss", loss, sync_dist=True)
        self.log("valid_acc", acc, sync_dist=True)

        if self.frame_pooling_enabled:
            y_hat = [torch.argmax(yi) for yi in y_hat]
            y_hat = y_hat[0]  # Only treat one feature for now
            y_hat = torch.argmax(torch.bincount(y_hat))
        return y, y_hat

    def validation_epoch_end(self, outputs):
        correct = 0
        label_correct = [0, 0, 0, 0]
        label_total = [0, 0, 0, 0]

        if self.frame_pooling_enabled:
            for y, y_hat in outputs:
                label_total[y] += 1
                if y == y_hat:
                    correct += 1
                    label_correct[y] += 1
        else:
            for y, y_hat in outputs:
                for y_item, y_hat_item in zip(y, y_hat):
                    label = y_item.item()
                    label_total[label] += 1
                    if torch.argmax(y_hat_item) == label:
                        correct += 1
                        label_correct[label] += 1

        label_acc = [label_correct[0] / label_total[0],
                     label_correct[1] / label_total[1],
                     label_correct[2] / label_total[2],
                     label_correct[3] / label_total[3]]

        ua = sum(label_acc) / len(label_acc)
        wa = correct / sum(label_total)
        acc = wa + ua
        self.log("wa", wa)
        self.log("ua", ua)
        self.log("sum_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.frame_pooling_enabled:
            y_hat = [torch.argmax(self(x), dim=1)]
        else:
            y_hat = [torch.argmax(self(xi), dim=1) for xi in x]
        y_hat = y_hat[0]  # Only treat one feature for now
        y_hat = torch.argmax(torch.bincount(y_hat))

        return y, y_hat

    def test_epoch_end(self, outputs):
        correct = 0
        label_correct = [0, 0, 0, 0]
        label_total = [0, 0, 0, 0]

        y_true = []
        y_pred = []

        for y, y_hat in outputs:
            y = y[0][0]
            y_true.append(y)
            y_pred.append(y_hat)
            label_total[y] += 1
            if y == y_hat:
                correct += 1
                label_correct[y] += 1

        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)

        label_acc = [label_correct[0] / label_total[0],
                     label_correct[1] / label_total[1],
                     label_correct[2] / label_total[2],
                     label_correct[3] / label_total[3]]

        ua = sum(label_acc) / len(label_acc)
        wa = correct / sum(label_total)
        self.log("wa", wa)
        self.log("ua", ua)
        cm = FM.confusion_matrix(preds=y_pred, target=y_true, num_classes=4)
        cm *= 100
        cm = cm.float()
        for i in range(4):
            cm[i] /= label_total[i]
        df_cm = pandas.DataFrame(cm.numpy(), index=['neutral', 'happy', 'sad', 'angry'], columns=['neutral', 'happy', 'sad', 'angry'])
        plt.figure()
        seaborn.heatmap(df_cm, annot=True, fmt='05.2f', cmap='YlGnBu').get_figure()

        i = 0
        while os.path.exists("result_cm_{}.png".format(str(i))):
            i += 1
        plt.savefig("result_cm_{}.png".format(str(i)))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    # test_x = torch.empty((100, 1, 125, 81))
    # test_y = CATP()(test_x)
    # print(test_y.shape)
    # y = torch.Tensor([[2]] * 100).long()
    # print(F.cross_entropy(test_y[0].unsqueeze(0), y[0]))
    model = CATP(out_types=4, dropout_rate=0.5, aacnn_mode=True, frame_pooling_enabled=False, target_dim=30, max_length=30 * 80)
    torchsummary.summary(model, (1, 128, 81))
