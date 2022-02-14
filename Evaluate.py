import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, label_ranking_average_precision_score, coverage_error, precision_recall_curve
import pickle
from scipy.stats import sem
import os
from copy import deepcopy as copy

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


def evaluate(obs_test, fitted_params, theta_true=None, p_true=None, print_res=False, one_epoch=False, F1_res=10):
    tabRes = []
    labs = []

    obs_test = copy(obs_test)

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
        ap = average_precision_score(true, pred, average="micro")

        rankAvgPrec = label_ranking_average_precision_score(true, pred)
        c=coverage_error(true, pred)
        covErrNorm = (c-1)/nbOut
        F1 = 0
        for thres in np.linspace(0, 1, F1_res):
            if F1_res==1:
                thres=0.5
            F1_tmp = f1_score(true, (np.array(pred)>thres).astype(int), average="micro")
            if F1_tmp > F1:
                F1 = F1_tmp

        mae = -1
        rmse = -1
        if theta_true is not None:
            diffTet = np.abs(theta-theta_true)
            mae = np.mean(diffTet)
            rmse = np.mean(diffTet**2)**0.5

        labs = ["roc", "F1", "ap", "mae", "rmse", "rankAvgPrec", "covErrNorm"]
        tabRes.append([roc, F1, ap, mae, rmse, rankAvgPrec, covErrNorm])

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

# Real world XP for every K
def XP4_allK(folder_base="XP/RW/", ds=None):

    if ds is None:
        listDs = [
                    "epigraphy_alt",
                    "lastfm",
                    "lastfm_alt",
                    "wikipedia",
                    "wikipedia_alt",
                    "reddit",
                    "reddit_alt",
                    "epigraphy",
                  ]
    else:
        listDs = [ds]


    for ds in listDs:
        print(ds)
        folder = folder_base+ds+"/"
        folderFig = folder.replace("XP", "Plots")
        ensureFolder(folderFig+"/")

        folds = 5
        codeSave = ds+"_"
        nbLoops = 1000
        log_beta_bb=(-1, 2)
        res_beta = 10
        if "epigraphy" in ds:
            res_beta = 100


        obs_train, obs_validation, obs_test, indt_to_time = getData(folder+"/", codeSave)

        tabx = []
        tabRes, tabRes_beta_null, tabRes_one_epoch = [], [], []
        tabStd, tabStd_beta_null, tabStd_one_epoch = [], [], []
        codeSaveFig = f"_"
        for K in [5, 10, 20, 30]:
            fitted_params = getParams(folder, codeSave+f"{K}_", folds)
            fitted_params_beta_null = getParams(folder, codeSave+f"{K}_"+"beta_null_", folds)
            fitted_params_one_epoch = getParams(folder, codeSave+f"{K}_"+"one_epoch_", folds)

            res_mean, res_std, res_sem = evaluate(obs_test, fitted_params, print_res=True, F1_res=50)
            tabRes.append(res_mean)
            tabStd.append(res_std)
            res_mean_beta_null, res_std_beta_null, res_sem_beta_null = evaluate(obs_test, fitted_params_beta_null, print_res=True, F1_res=50)
            tabRes_beta_null.append(res_mean_beta_null)
            tabStd_beta_null.append(res_std_beta_null)
            res_mean_one_epoch, res_std_one_epoch, res_sem_one_epoch = evaluate(obs_test, fitted_params_one_epoch, print_res=True, one_epoch=True, F1_res=50)
            tabRes_one_epoch.append(res_mean_one_epoch)
            tabStd_one_epoch.append(res_std_one_epoch)

            tabx.append(K)

        tabRes = np.array(tabRes)
        tabRes_beta_null = np.array(tabRes_beta_null)
        tabRes_one_epoch = np.array(tabRes_one_epoch)
        tabStd = np.array(tabStd)
        tabStd_beta_null = np.array(tabStd_beta_null)
        tabStd_one_epoch = np.array(tabStd_one_epoch)

        for metric in [(0, "AUC ROC"), (1, "F1 score"), (2, "Average precision"), (5, "Rank average precision"), (6, "Normalized coverage error")]:
            plt.plot(tabx, tabRes[:, metric[0]], "b", label="SDSBM")
            plt.fill_between(tabx, tabRes[:, metric[0]]-tabStd[:, metric[0]], tabRes[:, metric[0]]+tabStd[:, metric[0]], color="b", alpha=0.3)
            plt.plot(tabx, tabRes_beta_null[:, metric[0]], "r", label="No coupling")
            plt.fill_between(tabx, tabRes_beta_null[:, metric[0]]-tabStd_beta_null[:, metric[0]], tabRes_beta_null[:, metric[0]]+tabStd_beta_null[:, metric[0]], color="r", alpha=0.3)
            plt.plot(tabx, tabRes_one_epoch[:, metric[0]], "g", label="No temporal dependence")
            plt.fill_between(tabx, tabRes_one_epoch[:, metric[0]]-tabStd_one_epoch[:, metric[0]], tabRes_one_epoch[:, metric[0]]+tabStd_one_epoch[:, metric[0]], color="g", alpha=0.3)
            plt.xlabel(r"Number of clusters K")
            plt.ylabel(metric[1])
            plt.legend()
            plt.savefig(folderFig+codeSaveFig+metric[1]+"_vs_K.pdf")
            plt.close()


def alluvialPlot():

    import plotly
    from matplotlib.colors import to_rgba

    listDs = [
        "epigraphy",
        # "epigraphy_alt",
        # "lastfm",
        # "lastfm_alt",
        # "wikipedia",
        # "wikipedia_alt",
        # "reddit",
        # "reddit_alt",
    ]

    folder_base="XP/RW/"
    ds = "epigraphy"
    K = 5
    folds = 5
    fold = 0

    folder = folder_base+ds+"/"
    folderFig = folder.replace("XP", "Plots")
    ensureFolder(folderFig+"/")
    codeSave = ds+"_"



    fitted_params = getParams(folder, codeSave+f"{K}_", folds)

    with open("XP/RW/Data/epigraphy_indsToTitles.pkl", "rb") as f:
        ind_to_title = pickle.load(f)
    with open("XP/RW/Data/epigraphy_indsToRegions.pkl", "rb") as f:
        ind_to_region = pickle.load(f)
    with open("XP/RW/Data/epigraphy_tmin.pkl", "rb") as f:
        tmin = pickle.load(f)

    theta, p, beta = fitted_params[fold]
    I = len(theta[0])
    K = len(p)

    obs_train, obs_validation, obs_test, indt_to_time = getData(folder+"/", codeSave)
    weigthI = np.zeros((I))
    for (i,o,indt) in obs_train[fold]:
        weigthI[i] += 1
    print(weigthI)

    for k in range(len(p)):
        print(k, "=======")
        for val, o in reversed(sorted(zip(list(p[k]), list(range(len(p[k])))))):
            if val>0.05:
                print(val, ind_to_region[o])

    nomsClusters = ["Rome", "Italia", "Illyria, Hispania, Gauls", "Eastern Europe", "Germany, Asia"]

    c_items = []
    c_links = []
    groups = []
    opacity = 0.6
    opacity_nodes = 0.9
    thresPlot = 400
    colors = ["orange","y", 'navy', "darkgreen", "darkred", ]
    colors = ["orange","yellow", 'b', "g", "red", ]

    c_clus = ["rgba"+str(to_rgba("lightgray", opacity_nodes)) for k in range(K)]
    print(c_clus)
    toRem = []
    for i in range(I):
        k=-1
        if "servi" in ind_to_title[i]: k=0
        if "liberti" in ind_to_title[i]: k=1
        if "milit" in ind_to_title[i]: k=2
        if "senatorius" in ind_to_title[i]: k=3
        if "augusti" in ind_to_title[i]: k=4
        if k==-1: toRem.append(i)

        #k = np.argmax(np.mean(theta[:, i], axis=0))
        c = to_rgba(colors[k], opacity)
        c_links.append("rgba"+str(c))
        c = to_rgba(colors[k], opacity_nodes)
        c_items.append("rgba"+str(c))
        groups.append(k)


    labels = [ind_to_title[i].replace("_", " ").capitalize() for i in range(len(theta[0]))]+[fr"{nomsClusters[k]} · year {tmin}" for k in range(K)]
    color_nodes = [c_items[i] for i in range(len(theta[0]))]+[c_clus[k] for k in range(K)]
    print(labels)
    source, target, value = [], [], []
    color = []
    for i in range(I):
        if i in toRem: continue
        for k in range(K):
            if theta[0,i,k]*weigthI[i]>thresPlot:
                source.append(i)
                target.append(I+k)
                value.append(theta[0,i,k]*weigthI[i])
                color.append(c_links[i])

    indt = 0
    tetPrec = theta[0]
    for t in range(1, len(theta)):
        if t%100!=0:
            continue
        labels += [fr"{nomsClusters[k]} · year {tmin+t}" for k in range(K)]
        color_nodes += [c_clus[k] for k in range(K)]
        trans = np.zeros((K,K))
        for i in range(I):
            if i in toRem: continue
            trans = np.zeros((K,K))
            loss = theta[t,i]-tetPrec[i]  # k
            for k in range(K):

                div = copy(loss)
                div[div<0]=0

                val = -loss[k]*div/np.sum(div)
                val[val<0] = 0.

                trans[k] += val

                if loss[k]<0:
                    trans[k,k] += theta[t,i,k]
                else:
                    trans[k,k] += tetPrec[i,k]

            for k1 in range(K):
                for k2 in range(K):
                    if trans[k1,k2]*weigthI[i]>thresPlot:
                        source.append(I+K*(indt)+k1)
                        target.append(I+K*(indt+1)+k2)
                        value.append(trans[k1,k2]*weigthI[i])
                        color.append(c_links[i])

        tetPrec = theta[t]
        indt += 1

    labels += [ind_to_title[i].replace("_", " ").capitalize()+" " for i in range(len(theta[0]))]
    color_nodes += [c_items[i] for i in range(len(theta[0]))]
    for i in range(I):
        if i in toRem: continue
        for k in range(K):
            if tetPrec[i,k]*weigthI[i]>thresPlot:
                source.append(I+K*(indt)+k)
                target.append(I+K*(indt+1)+i)
                value.append(tetPrec[i,k]*weigthI[i])
                color.append(c_links[i])

    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.3),
            label = labels,
            color = color_nodes,
        ),
        link = dict(
            source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
            target = target,
            value = value,
            color = color,
        ))])

    fig.update_layout(title_text="Status geographic evolution from latin graves (100BC - 500AC)", font_size=13, font_family="Serif", font_color="black")
    fig.write_image("Plots/RW/Status.pdf", height=1080, width=1920, scale=2)
    #fig.show()

# Varying Nepochs
def IllustrationMethod(folder = "XP/Synth/Nepochs/"):
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

    scale=7
    plt.figure(figsize=(0.60*scale, 0.75*scale))
    for type_i, typeVar in enumerate(["sin", "rnd"]):
        plt.subplot(2,1,type_i+1)
        for Nepochs_div in reversed([5]):  # = Nobs moyen par epoque
            Nepochs = int(NobsperI/Nepochs_div)
            codeSave = f"{typeVar}_Nepochs={Nepochs}_"

            theta_true, p_true = getTrueParams(folder, codeSave)

            fitted_params = getParams(folder, codeSave, folds)
            fitted_params_beta_null = getParams(folder, codeSave+"beta_null_", folds)
            fitted_params_one_epoch = getParams(folder, codeSave+"one_epoch_", folds)

            arrx = np.array(list(range(len(theta_true))))
            i = 0
            fold = 0

            for k in range(3):
                if k==0:
                    plt.plot(fitted_params_beta_null[fold][0][:, i, k], "b-", alpha=0.3, label="Independent time slices")
                    plt.plot(arrx, [fitted_params_one_epoch[fold][0][:, i, k]]*len(arrx), "y-", label="Static SBM")
                    plt.plot(fitted_params[fold][0][:, i, k], "r-", label="SDSBM")
                    plt.plot(theta_true[:, i, k], "k-", linewidth=2, label="Ground truth")
                else:
                    plt.plot(fitted_params_beta_null[fold][0][:, i, k], "b-", alpha=0.3)
                    plt.plot(arrx, [fitted_params_one_epoch[fold][0][:, i, k]]*len(arrx), "y-")
                    plt.plot(fitted_params[fold][0][:, i, k], "r-")
                    plt.plot(theta_true[:, i, k], "k-", linewidth=2)
            plt.ylim([-0.05,1.05])
            plt.xticks([])
            if type_i!=0:
                plt.xlabel("Time")
                plt.legend()
            plt.ylabel("Memberships")

    plt.tight_layout()
    plt.savefig("Plots/Illustration.pdf")




IllustrationMethod()
pause()
XP = input("What to evaluate > ")
if XP=="123":
    XP1()
    XP2()
    XP3()
else:
    XP4_allK(ds=XP)

alluvialPlot()