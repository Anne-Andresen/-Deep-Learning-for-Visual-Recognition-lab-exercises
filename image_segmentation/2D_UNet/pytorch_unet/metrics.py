import torch
import torch.nn as nn
from torch.nn.functional import  cross_entropy
from torch.nn.modules.loss import _WeightedLoss

epsilon = 1e-32

class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None, ignore_index=-100):
        super(LogNLLLoss).__init__(weight, size_average, reduce, reduction)
        self.ignore_index =ignore_index

    def forward(self, y_label, y_target):
        y_label = torch.log(y_label + epsilon)
        return cross_entropy(y_label, y_target, weight=self.weight, ignore_index=self.ignore_index)


def classwise_iou(pred, label):

    dims = (0, *range(2, len(pred.shape)))
    label = torch.zeros_like(pred).scatter_(1, label[:, None, :], 1)
    intersection = pred*label
    union = pred + label - intersection
    classwise_iou = (intersection.sum(dim=dims).float() + epsilon) / (union.sum(dim=dims) + epsilon)

    return classwise_iou

def classwise_f1(pred, label):

    epsilon = 1e-20
    n_classes = pred.shape[1]

    pred = torch.argmax(pred, dim=1)
    true_positives = torch.tensor([((pred==i) * (label==i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(pred == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(label==i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon)/(selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    f1 = 2*(precision *recall) / (precision+recall)
    return f1

def class_weighted_metric(classwise_metric):

    def weighted_metric(pred, label, weights=None):
        dims = (0, *range(2, len(pred.shape)))

        if weights==None:
            weights = torch.ones(pred.shape[1]) / pred.shape[1]
        else:
            if len(weights)!= pred.shape[1]:
                raise  ValueError('# of weights must match with # of classes')
            if not  isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)

            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(pred, label).cpu()
        return (classwise_scores * weights).sum().item()
    return weighted_metric


f1_score = class_weighted_metric(classwise_f1)
jaccard = class_weighted_metric(classwise_iou)

if __name__=='__main__':
    pred, label = torch.zeros(3, 2, 5, 5), torch.zeros(3, 5, 5).long()
    print(classwise_iou(pred, label))



