import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import pickle
from scipy.stats import sem

def getData(folder, codeSave):
    with open(folder+codeSave+"obs_train.pkl", "rb") as f:
        obs_train = pickle.load(f)
    with open(folder+codeSave+"obs_validation.pkl", "rb") as f:
        obs_validation = pickle.load(f)
    with open(folder+codeSave+"obs_test.pkl", "rb") as f:
        obs_test = pickle.load(f)
    with open(folder+codeSave+"indt_to_time.pkl", "rb") as f:
        indt_to_time = pickle.load(f)

    return obs_train, obs_validation, obs_test, indt_to_time

def getTrueParams(folder, codeSave):
    theta = np.load(folder+codeSave+f"theta_true.npy")
    p = np.load(folder+codeSave+f"p_true.npy")

    return theta, p

def getParams(folder, codeSave, folds):
    fitted_params = []
    for fold in range(folds):
        theta = np.load(folder+codeSave+f"f{fold}_theta.npy")
        p = np.load(folder+codeSave+f"f{fold}_p.npy")
        beta = np.load(folder+codeSave+f"f{fold}_beta.npy")
        fitted_params.append((theta, p, beta))

    return fitted_params


def evaluate(obs_test, fitted_params, theta_true, p_true, print_res=False, one_epoch=False):
    tabRes = []
    labs = []

    if one_epoch:
        for fold in range(len(obs_test)):
            for i, (item,o,indt) in enumerate(obs_test[fold]):
                obs_test[fold][i] = (item,o,0)

    for fold in range(len(obs_test)):
        theta, p, beta = fitted_params[fold]

        pred, true = [], []
        nbOut = p.shape[-1]

        for (i,o,indt) in obs_test[fold]:
            true_tmp = np.zeros((nbOut))
            true_tmp[o] = 1

            pred_tmp = theta[indt,i].dot(p)

            true.append(true_tmp)
            pred.append(pred_tmp)

        roc = roc_auc_score(true, pred, average="micro")
        F1 = f1_score(true, (np.array(pred)>0.5).astype(int), average="micro")
        ap = average_precision_score(true, pred, average="micro")

        diffTet = np.abs(theta-theta_true)
        mae = np.mean(diffTet)
        rmse = np.mean(diffTet**2)**0.5

        labs = ["roc", "F1", "ap", "mae", "rmse"]
        tabRes.append([roc, F1, ap, mae, rmse])

        if fold==0 and False:
            for i in range(9):
                plt.subplot(3,3,i+1)
                if not one_epoch:
                    plt.plot(theta[:, i, :])
                else:
                    for k in range(theta.shape[-1]):
                        plt.plot([0, len(theta_true)], [theta[0, i, k], theta[0, i, k]])

                plt.plot(theta_true[:, i, :], "k")
            plt.show()

    tabRes = np.array(tabRes)
    res_mean = np.mean(tabRes, axis=0)
    res_std = np.std(tabRes, axis=0)
    res_sem = sem(tabRes, axis=0)

    if print_res:
        # print("\t".join(map(str, labs)))
        resstr = "\t".join([fr"{np.round(r, 4)} ± {np.round(e, 4)}" for r, e in zip(res_mean, res_std)])
        print(resstr.expandtabs(20))
        # print("\t".join(map(str, res_std)))
        # print("\t".join(map(str, res_sem)))

    return tabRes

def XP1(folder = "XP/Synth/NobsperI/"):
    I = 100
    K = 3
    O = 3

    Nepochs = 100
    Tmax = 2*np.pi
    nbLoops = 1000
    folds = 5
    res_beta = 40

    for typeVar in ["rnd", "sin"]:
        #for NobsperI in np.linspace(Nepochs, Nepochs*100, 21):  # =========================
        for NobsperI in np.linspace(Nepochs, Nepochs*10, 5):
            NobsperI = int(NobsperI)
            codeSave = f"{typeVar}_Nobs={NobsperI}_"
            print(codeSave)
            obs_train, obs_validation, obs_test, indt_to_time = getData(folder, codeSave)
            theta_true, p_true = getTrueParams(folder, codeSave)

            fitted_params = getParams(folder, codeSave, folds)
            fitted_params_beta_null = getParams(folder, codeSave+"beta_null_", folds)
            fitted_params_one_epoch = getParams(folder, codeSave+"one_epoch_", folds)

            tabRes = evaluate(obs_test, fitted_params, theta_true, p_true, print_res=True)
            tabRes = evaluate(obs_test, fitted_params_beta_null, theta_true, p_true, print_res=True)
            tabRes = evaluate(obs_test, fitted_params_one_epoch, theta_true, p_true, print_res=True, one_epoch=True)

XP1()