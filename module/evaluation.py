import torch
import numpy as np
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.nn.functional as F


def argmax_prediction(pred_y, true_y):
    pred_argmax = torch.argmax(pred_y, dim=1).cpu()
    true_y = true_y.cpu()
    return pred_argmax, true_y


def threshold_prediction(pred_y, true_y):
    pred_y = pred_y > 0.5
    return pred_y, true_y


def metrics_report(pred_y, true_y, label, get_dict=False, multilabel=False):
    if multilabel:
        pred_y, true_y = threshold_prediction(pred_y, true_y)
        available_label = sorted(list(set((pred_y == True).nonzero()[:, -1].tolist() + (true_y == True).nonzero()[:, -1].tolist())))
    else:
        pred_y, true_y = argmax_prediction(pred_y, true_y)
        available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

    class_name = list(label[available_label])
    if get_dict:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4, output_dict=True)
    else:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)


def metrics_report_for_emo_binary(pred_y, true_y, get_dict=False, multilabel=False):
    if multilabel:
        pred_y, true_y = threshold_prediction(pred_y, true_y)
        available_label = sorted(list(set((pred_y == True).nonzero()[:, -1].tolist() + (true_y == True).nonzero()[:, -1].tolist())))
    else:
        pred_y, true_y = argmax_prediction(pred_y, true_y)
        available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

    class_name = ['non-neutral', 'neutral']
    pred_y = [1 if element == 6 else 0 for element in pred_y] # element 6 means neutral.
    true_y = [1 if element == 6 else 0 for element in true_y]

    if get_dict:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4, output_dict=True)
    else:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)

def log_metrics(logger, emo_pred_y_list, emo_true_y_list, cau_pred_y_list, cau_true_y_list, cau_pred_y_list_all, cau_true_y_list_all, loss_avg, n_cause, option='train'):
    label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    logger.info('\n' + metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_))
    report_dict = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=True)
    acc_emo, p_emo, r_emo, f1_emo = report_dict['accuracy'], report_dict['weighted avg']['precision'], report_dict['weighted avg']['recall'], report_dict['weighted avg']['f1-score']
    logger.info(f'\nemotion: {option} | loss {loss_avg}\n')

    logger.info('\n' + metrics_report_for_emo_binary(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list)))
    report_dict = metrics_report_for_emo_binary(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), get_dict=True)
    acc_emo, p_emo, r_emo, f1_emo = report_dict['accuracy'], report_dict['weighted avg']['precision'], report_dict['weighted avg']['recall'], report_dict['weighted avg']['f1-score']
    logger.info(f'\nemotion (binary): {option} | loss {loss_avg}\n')

    if n_cause == 2:
        label_ = np.array(['No Cause', 'Cause'])

        report_dict = metrics_report(torch.cat(cau_pred_y_list), torch.cat(cau_true_y_list), label=label_, get_dict=True)
        _, p_cau, _, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']

        report_dict = metrics_report(torch.cat(cau_pred_y_list_all), torch.cat(cau_true_y_list_all), label=label_, get_dict=True)
        acc_cau, _, r_cau, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']

        f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
        logger.info(f'\nbinary_cause: {option} | loss {loss_avg}\n')
        logger.info(f'\nbinary_cause: accuracy: {acc_cau} | precision: {p_cau} | recall: {r_cau} | f1-score: {f1_cau}\n')
    else:
        label_ = np.array(['no-context', 'inter-personal', 'self-contagion', 'no cause'])
        report_dict = metrics_report(torch.cat(cau_pred_y_list), torch.cat(cau_true_y_list), label=label_, get_dict=True)

        p_no_context, p_inter_personal, p_self_contagion = report_dict['no-context']['precision'], report_dict['inter-personal']['precision'], report_dict['self-contagion']['precision']

        p_cau = (report_dict['no-context']['support'] * report_dict['no-context']['precision'] + report_dict['inter-personal']['support'] * report_dict['inter-personal']['precision'] + report_dict['self-contagion']['support'] * report_dict['self-contagion']['precision']) / \
                (report_dict['no-context']['support'] + report_dict['inter-personal']['support'] + report_dict['self-contagion']['support'] )

        report_dict = metrics_report(torch.cat(cau_pred_y_list_all), torch.cat(cau_true_y_list_all), label=label_, get_dict=True)

        r_no_context, r_inter_personal, r_self_contagion = report_dict['no-context']['recall'], report_dict['inter-personal']['recall'], report_dict['self-contagion']['recall']

        acc_cau = report_dict['accuracy']
        r_cau = (report_dict['no-context']['support'] * report_dict['no-context']['recall'] + report_dict['inter-personal']['support'] * report_dict['inter-personal']['recall'] + report_dict['self-contagion']['support'] * report_dict['self-contagion']['recall']) / \
                (report_dict['no-context']['support'] + report_dict['inter-personal']['support'] + report_dict['self-contagion']['support'] )

        f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
        logger.info(f'\nmulticlass_cause: {option} | loss {loss_avg}\n')

        logger.info(f'\nmulticlass_cause: no-context    | precision: {p_no_context} | recall: {r_no_context} | f1-score: {2 * p_no_context * r_no_context / (p_no_context + r_no_context) if p_no_context + r_no_context != 0 else 0}\n')
        logger.info(f'multiclass_cause: inter-personal  | precision: {p_inter_personal} | recall: {r_inter_personal} | f1-score: {2 * p_inter_personal * r_inter_personal / (p_inter_personal + r_inter_personal) if p_inter_personal + r_inter_personal != 0 else 0}\n')
        logger.info(f'multiclass_cause: self-contagion  | precision: {p_self_contagion} | recall: {r_self_contagion} | f1-score: {2 * p_self_contagion * r_self_contagion / (p_self_contagion + r_self_contagion) if p_self_contagion + r_self_contagion != 0 else 0}\n')

        logger.info(f'\nmulticlass_cause: accuracy: {acc_cau} | precision: {p_cau} | recall: {r_cau} | f1-score: {f1_cau}\n')

    return p_cau, r_cau, f1_cau



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
