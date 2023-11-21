from torch.nn import functional as F
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(dataset, model):
    prob = []
    with torch.no_grad():
        for data in dataset:
            inputs = model.processor(data["img"], return_tensors="pt").to(model.device)
            logits = model.predictor(**inputs).logits.unsqueeze(0)
            prob.append(F.softmax(logits, dim=-1))
    return torch.cat(prob)


# https://arxiv.org/abs/2205.08096
def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    print("getting retain prob")
    retain_prob = collect_prob(retain_loader, model)
    print("getting forget prob")
    forget_prob = collect_prob(forget_loader, model)
    print("getting test prob")
    test_prob = collect_prob(test_loader, model)

    print(retain_prob.shape, forget_prob.shape, test_prob.shape)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


# https://arxiv.org/abs/2205.08096
def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )
    # MIA with SVC
    clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    results_svc = results.mean()

    # MIA with Logistic Regression
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    results_lr = results.mean()
    return results_svc, results_lr

