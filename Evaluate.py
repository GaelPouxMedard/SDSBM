import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import pickle
from scipy.stats import sem
import os

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

    return res_mean, res_std, res_sem

def ensureFolder(folder):
    curfol = "./"
    for fol in folder.split("/"):
        if fol not in os.listdir(curfol) and fol!="":
            os.mkdir(curfol+fol)
        curfol += fol+"/"

# Varying NobsperI
def XP1(folder = "XP/Synth/NobsperI/"):
    folderFig = folder.replace("XP", "Plots")
    ensureFolder(folderFig)
    I = 100
    K = 3
    O = 3

    Nepochs = 100
    Tmax = 2*np.pi
    nbLoops = 1000
    folds = 5
    res_beta = 40

    for typeVar in ["rnd", "sin"]:
        tabx = []
        tabRes, tabRes_beta_null, tabRes_one_epoch = [], [], []
        tabStd, tabStd_beta_null, tabStd_one_epoch = [], [], []
        codeSaveFig = f"{typeVar}_"

        for NobsperI in np.linspace(Nepochs, Nepochs*100, 21):
            NobsperI = int(NobsperI)
            codeSave = f"{typeVar}_Nobs={NobsperI}_"
            print(codeSave)
            obs_train, obs_validation, obs_test, indt_to_time = getData(folder, codeSave)
            theta_true, p_true = getTrueParams(folder, codeSave)

            fitted_params = getParams(folder, codeSave, folds)
            fitted_params_beta_null = getParams(folder, codeSave+"beta_null_", folds)
            fitted_params_one_epoch = getParams(folder, codeSave+"one_epoch_", folds)

            res_mean, res_std, res_sem = evaluate(obs_test, fitted_params, theta_true, p_true, print_res=True)
            tabRes.append(res_mean)
            tabStd.append(res_std)
            res_mean_beta_null, res_std_beta_null, res_sem_beta_null = evaluate(obs_test, fitted_params_beta_null, theta_true, p_true, print_res=True)
            tabRes_beta_null.append(res_mean_beta_null)
            tabStd_beta_null.append(res_std_beta_null)
            res_mean_one_epoch, res_std_one_epoch, res_sem_one_epoch = evaluate(obs_test, fitted_params_one_epoch, theta_true, p_true, print_res=True, one_epoch=True)
            tabRes_one_epoch.append(res_mean_one_epoch)
            tabStd_one_epoch.append(res_std_one_epoch)

            tabx.append(NobsperI)

        tabRes = np.array(tabRes)
        tabRes_beta_null = np.array(tabRes_beta_null)
        tabRes_one_epoch = np.array(tabRes_one_epoch)
        tabStd = np.array(tabStd)
        tabStd_beta_null = np.array(tabStd_beta_null)
        tabStd_one_epoch = np.array(tabStd_one_epoch)
        
        for metric in [(0, "AUC ROC"), (4, "RMSE")]:
            plt.plot(tabx, tabRes[:, metric[0]], "b", label="SDSBM")
            plt.fill_between(tabx, tabRes[:, metric[0]]-tabStd[:, metric[0]], tabRes[:, metric[0]]+tabStd[:, metric[0]], color="b", alpha=0.3)
            plt.plot(tabx, tabRes_beta_null[:, metric[0]], "r", label="No coupling")
            plt.fill_between(tabx, tabRes_beta_null[:, metric[0]]-tabStd_beta_null[:, metric[0]], tabRes_beta_null[:, metric[0]]+tabStd_beta_null[:, metric[0]], color="r", alpha=0.3)
            plt.plot(tabx, tabRes_one_epoch[:, metric[0]], "g", label="No temporal dependence")
            plt.fill_between(tabx, tabRes_one_epoch[:, metric[0]]-tabStd_one_epoch[:, metric[0]], tabRes_one_epoch[:, metric[0]]+tabStd_one_epoch[:, metric[0]], color="g", alpha=0.3)
            plt.xlabel("Number of observations per item")
            plt.ylabel(metric[1])
            if metric[1]=="RMSE": plt.ylim([0, 0.45])
            if metric[1]=="AUC ROC": plt.ylim([0.5, 0.85])
            plt.legend()
            plt.savefig(folderFig+codeSaveFig+metric[1]+".pdf")
            plt.close()

# Varying Nepochs
def XP2(folder = "XP/Synth/Nepochs/"):
    folderFig = folder.replace("XP", "Plots")
    ensureFolder(folderFig)
    I = 100
    K = 3
    O = 3

    NobsperI = 1000
    Tmax = 2*np.pi
    nbLoops = 1000
    folds = 5
    res_beta = 40

    for typeVar in ["sin", "rnd"]:
        tabx = []
        tabRes, tabRes_beta_null, tabRes_one_epoch = [], [], []
        tabStd, tabStd_beta_null, tabStd_one_epoch = [], [], []
        codeSaveFig = f"{typeVar}_"
        for Nepochs_div in reversed([1, 5, 10, 20, 30, 40, 50, 75, 100]):  # = Nobs moyen par epoque
            Nepochs = int(NobsperI/Nepochs_div)
            codeSave = f"{typeVar}_Nepochs={Nepochs}_"
            print(codeSave)
            obs_train, obs_validation, obs_test, indt_to_time = getData(folder, codeSave)
            theta_true, p_true = getTrueParams(folder, codeSave)

            fitted_params = getParams(folder, codeSave, folds)
            fitted_params_beta_null = getParams(folder, codeSave+"beta_null_", folds)
            fitted_params_one_epoch = getParams(folder, codeSave+"one_epoch_", folds)

            res_mean, res_std, res_sem = evaluate(obs_test, fitted_params, theta_true, p_true, print_res=True)
            tabRes.append(res_mean)
            tabStd.append(res_std)
            res_mean_beta_null, res_std_beta_null, res_sem_beta_null = evaluate(obs_test, fitted_params_beta_null, theta_true, p_true, print_res=True)
            tabRes_beta_null.append(res_mean_beta_null)
            tabStd_beta_null.append(res_std_beta_null)
            res_mean_one_epoch, res_std_one_epoch, res_sem_one_epoch = evaluate(obs_test, fitted_params_one_epoch, theta_true, p_true, print_res=True, one_epoch=True)
            tabRes_one_epoch.append(res_mean_one_epoch)
            tabStd_one_epoch.append(res_std_one_epoch)

            tabx.append(Nepochs)

        tabRes = np.array(tabRes)
        tabRes_beta_null = np.array(tabRes_beta_null)
        tabRes_one_epoch = np.array(tabRes_one_epoch)
        tabStd = np.array(tabStd)
        tabStd_beta_null = np.array(tabStd_beta_null)
        tabStd_one_epoch = np.array(tabStd_one_epoch)

        for metric in [(0, "AUC ROC"), (4, "RMSE")]:
            plt.plot(tabx, tabRes[:, metric[0]], "b", label="SDSBM")
            plt.fill_between(tabx, tabRes[:, metric[0]]-tabStd[:, metric[0]], tabRes[:, metric[0]]+tabStd[:, metric[0]], color="b", alpha=0.3)
            plt.plot(tabx, tabRes_beta_null[:, metric[0]], "r", label="No coupling")
            plt.fill_between(tabx, tabRes_beta_null[:, metric[0]]-tabStd_beta_null[:, metric[0]], tabRes_beta_null[:, metric[0]]+tabStd_beta_null[:, metric[0]], color="r", alpha=0.3)
            plt.plot(tabx, tabRes_one_epoch[:, metric[0]], "g", label="No temporal dependence")
            plt.fill_between(tabx, tabRes_one_epoch[:, metric[0]]-tabStd_one_epoch[:, metric[0]], tabRes_one_epoch[:, metric[0]]+tabStd_one_epoch[:, metric[0]], color="g", alpha=0.3)
            plt.xlabel("Number of epochs")
            plt.ylabel(metric[1])
            if metric[1]=="RMSE": plt.ylim([0, 0.45])
            if metric[1]=="AUC ROC": plt.ylim([0.5, 0.85])
            plt.legend()
            plt.savefig(folderFig+codeSaveFig+metric[1]+".pdf")
            plt.close()

# Varying p
def XP3(folder = "XP/Synth/VarP/"):
    folderFig = folder.replace("XP", "Plots")
    ensureFolder(folderFig)
    I = 100
    K = 3
    O = 3

    NobsperI = 1000
    Nepochs = 100
    Tmax = 2*np.pi
    nbLoops = 1000
    folds = 5
    res_beta = 40

    for typeVar in ["sin", "rnd"]:
        for use_p_true in [True, False]:
            tabx = []
            tabRes, tabRes_beta_null, tabRes_one_epoch = [], [], []
            tabStd, tabStd_beta_null, tabStd_one_epoch = [], [], []
            codeSaveFig = f"{typeVar}_inferp={use_p_true}_"
            for shiftp in np.linspace(0, 0.5, 21):
                codeSave = f"{typeVar}_shiftp={round(shiftp, 4)}_inferp={use_p_true}_"
                print(codeSave)
                obs_train, obs_validation, obs_test, indt_to_time = getData(folder, codeSave)
                theta_true, p_true = getTrueParams(folder, codeSave)

                fitted_params = getParams(folder, codeSave, folds)
                fitted_params_beta_null = getParams(folder, codeSave+"beta_null_", folds)
                fitted_params_one_epoch = getParams(folder, codeSave+"one_epoch_", folds)

                res_mean, res_std, res_sem = evaluate(obs_test, fitted_params, theta_true, p_true, print_res=True)
                tabRes.append(res_mean)
                tabStd.append(res_std)
                res_mean_beta_null, res_std_beta_null, res_sem_beta_null = evaluate(obs_test, fitted_params_beta_null, theta_true, p_true, print_res=True)
                tabRes_beta_null.append(res_mean_beta_null)
                tabStd_beta_null.append(res_std_beta_null)
                res_mean_one_epoch, res_std_one_epoch, res_sem_one_epoch = evaluate(obs_test, fitted_params_one_epoch, theta_true, p_true, print_res=True, one_epoch=True)
                tabRes_one_epoch.append(res_mean_one_epoch)
                tabStd_one_epoch.append(res_std_one_epoch)

                entropy_p = np.mean(np.sum(-p_true*np.log(p_true+1e-20), axis=1))
                tabx.append(entropy_p)

            tabRes = np.array(tabRes)
            tabRes_beta_null = np.array(tabRes_beta_null)
            tabRes_one_epoch = np.array(tabRes_one_epoch)
            tabStd = np.array(tabStd)
            tabStd_beta_null = np.array(tabStd_beta_null)
            tabStd_one_epoch = np.array(tabStd_one_epoch)

            for metric in [(0, "AUC ROC"), (4, "RMSE")]:
                plt.plot(tabx, tabRes[:, metric[0]], "b", label="SDSBM")
                plt.fill_between(tabx, tabRes[:, metric[0]]-tabStd[:, metric[0]], tabRes[:, metric[0]]+tabStd[:, metric[0]], color="b", alpha=0.3)
                plt.plot(tabx, tabRes_beta_null[:, metric[0]], "r", label="No coupling")
                plt.fill_between(tabx, tabRes_beta_null[:, metric[0]]-tabStd_beta_null[:, metric[0]], tabRes_beta_null[:, metric[0]]+tabStd_beta_null[:, metric[0]], color="r", alpha=0.3)
                plt.plot(tabx, tabRes_one_epoch[:, metric[0]], "g", label="No temporal dependence")
                plt.fill_between(tabx, tabRes_one_epoch[:, metric[0]]-tabStd_one_epoch[:, metric[0]], tabRes_one_epoch[:, metric[0]]+tabStd_one_epoch[:, metric[0]], color="g", alpha=0.3)
                plt.xlabel(r"Entropy of $p$")
                plt.ylabel(metric[1])
                if metric[1]=="RMSE": plt.ylim([0, 0.45])
                if metric[1]=="AUC ROC": plt.ylim([0.5, 0.85])
                plt.legend()
                plt.savefig(folderFig+codeSaveFig+metric[1]+".pdf")
                plt.close()

# Real world XP
def XP4(folder="XP/RW/", ds="lastfm"):

    listDs = ["epigraphy",
              "epigraphy_alt",
              "lastfm",
              "lastfm_alt",
              "wikipedia",
              "wikipedia_alt",
              "reddit",
              "reddit_alt",
              ]

    for ds in listDs:
        ensureFolder(folder+ds+"/")

        folds = 5
        codeSave = ds+"_"
        nbLoops = 1000
        log_beta_bb=(-1, 2)
        res_beta = 10
        if "epigraphy" in ds:
            res_beta = 100


        obs, indt_to_time = getDataRW(folder, ds)
        obs_train, obs_validation, obs_test = splitDS(obs, folds)
        saveData(folder+f"{ds}/", codeSave, obs_train, obs_validation, obs_test, indt_to_time)

        for K in [5, 10, 20, 30]:
            tic = time.time()
            fitted_params = run(copy(obs_train), copy(obs_validation), K, indt_to_time, nbLoops=nbLoops, log_beta_bb=log_beta_bb, res_beta=res_beta, use_p_true=False, printProg=True, rw=True)
            saveParams(folder+f"{ds}/", codeSave+f"{K}_", fitted_params)
            fitted_params = run(copy(obs_train), copy(obs_validation), K, indt_to_time, nbLoops=nbLoops, set_beta_null=True, use_p_true=False, printProg=True, rw=True)
            saveParams(folder+f"{ds}/", codeSave+f"{K}_"+"beta_null_", fitted_params)
            fitted_params = run(copy(obs_train), copy(obs_validation), K, indt_to_time, nbLoops=nbLoops, one_epoch=True, use_p_true=False, printProg=True, rw=True)
            saveParams(folder+f"{ds}/", codeSave+f"{K}_"+"one_epoch_", fitted_params)
            print(f"K={K} - {np.round((time.time()-tic)/(3600), 2)}h elapsed =====================================")


XP3()