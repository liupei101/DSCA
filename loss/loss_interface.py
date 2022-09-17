import torch
import torch.nn as nn


def create_survloss(loss, argv):
    assert loss in ['survple', 'survmle'], 'Unexpected loss. Only supprt survple and survmle.'
    if loss == 'survmle':
        return SurvMLE(**argv)
    elif loss == 'survple':
        return SurvPLE()


class SurvMLE(nn.Module):
    """A maximum likelihood estimation function in Survival Analysis.

    As suggested in '10.1109/TPAMI.2020.2979450',
        [*] L = (1 - alpha) * loss_l + alpha * loss_z.
    where loss_l is the negative log-likelihood loss, loss_z is an upweighted term for instances 
    D_uncensored. In discrete model, T = 0 if t in [0, a_1), T = 1 if t in [a_1, a_2) ...

    This implementation is based on https://github.com/mahmoodlab/MCAT/blob/master/utils/utils.py
    """
    def __init__(self, alpha=0.0, eps=1e-7):
        super(SurvMLE, self).__init__()
        self.alpha = alpha
        self.eps = eps
        print('[setup] loss: a MLE loss in discrete SA models with alpha = %.2f' % self.alpha)

    def forward(self, y, hazards_hat, cur_alpha=None):
        """
        y: torch.FloatTensor() with shape of [B, 2] for a discrete model.
        t: torch.LongTensor() with shape of [B, ] or [B, 1]. It's a discrete time label.
        c: torch.FloatTensor() with shape of [B, ] or [B, 1]. 
            c = 0 for uncensored samples (with event), 
            c = 1 for censored samples (without event).
        hazards_hat: torch.FloatTensor() with shape of [B, MAX_T]
        """
        t, c = y[:, 0], y[:, 1]
        batch_size = len(t)
        t = t.view(batch_size, 1).long() # ground truth bin, 0 [0,a_1), 1 [a_1,a_2),...,k-1 [a_k-1,inf)
        c = c.view(batch_size, 1).float() # censorship status, 0 or 1
        S = torch.cumprod(1 - hazards_hat, dim=1) # surival is cumulative product of 1 - hazards
        S_padded = torch.cat([torch.ones_like(c), S], 1) # s[0] = 1.0 to avoid for t = 0
        uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, t).clamp(min=self.eps)) + torch.log(torch.gather(hazards_hat, 1, t).clamp(min=self.eps)))
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, t+1).clamp(min=self.eps))
        neg_l = censored_loss + uncensored_loss
        alpha = self.alpha if cur_alpha is None else cur_alpha
        loss = (1.0 - alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        
        return loss
        

class SurvPLE(nn.Module):
    """A partial likelihood estimation (called Breslow estimation) function in Survival Analysis.

    This is a pytorch implementation by Huang. See more in https://github.com/huangzhii/SALMON.
    Note that it only suppurts survival data with no ties (i.e., event occurrence at same time).
    
    Args:
        y (Tensor): The absolute value of y indicates the last observed time. The sign of y 
        represents the censor status. Negative value indicates a censored example.
        y_hat (Tensor): Predictions given by the survival prediction model.
    """
    def __init__(self):
        super(SurvPLE, self).__init__()
        print('[setup] loss: a popular PLE loss in coxph')

    def forward(self, y, y_hat):
        device = y_hat.device

        T = torch.abs(y)
        E = (y > 0).int()

        n_batch = len(T)
        R_matrix_train = torch.zeros([n_batch, n_batch], dtype=torch.int8)
        for i in range(n_batch):
            for j in range(n_batch):
                R_matrix_train[i, j] = T[j] >= T[i]

        train_R = R_matrix_train.float().to(device)
        train_ystatus = E.float().to(device)

        theta = y_hat.reshape(-1)
        exp_theta = torch.exp(theta)

        loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

        return loss_nn

def loss_reg_l1(coef):
    print('[setup] L1 loss with coef={}'.format(coef))
    coef = .0 if coef is None else coef
    def func(model_params):
        if coef <= 1e-8:
            return 0.0
        else:
            return coef * sum([torch.abs(W).sum() for W in model_params])
    return func
