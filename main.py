import pickle

import numpy as np
from copy import deepcopy as copy
import sys
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.special import gamma, gammaln
import os
import pprofile
import time

"""
import pprofile
profiler = pprofile.Profile()
with profiler:
    RhovsT(g, 0.1, 0.28999, 1, 1000)
profiler.print_stats()
profiler.dump_stats("Benchmark.txt")
pause()
"""

def getData(I,O,K,Nepochs, NobsperI,Tmax, typeVar="sin", shiftp=0.05):
    p = np.random.random((K, O))
    p /= p.sum(-1)[:, None]

    p = np.array([[1.-shiftp, shiftp, 0.0],
                 [0.0, 1.-shiftp, shiftp],
                 [shiftp, 0.0, 1.-shiftp]],)

    obs = []
    clock = 0
    indt = -1
    indt_to_time = {}
    theta_t = []

    if typeVar=="sin":
        rnd_num = np.random.random((I, K))
        ftet = lambda x: np.array([[np.cos(rnd_num[i,k]+x+np.pi*k/K)**2 for k in range(K)] for i in range(I)])

    elif typeVar=="rnd":
        rnd_num = np.random.random((I, K))
        rnd_slope = ((np.random.random((I,K))<0.5).astype(int)-0.5)*0.5
        ftet = lambda x: np.array([rnd_num[i] if x<Tmax/4 else np.abs(rnd_num[i]+(x-Tmax/4)*rnd_slope[i]) for i in range(I)])


    ftetnorm = lambda x: ftet(x)/ftet(x).sum(-1)[:, None]
    theta = ftetnorm(clock)

    epoch = 0
    for nobs in range(NobsperI):
        if clock >= epoch - 1e-10:
            indt += 1
            indt_to_time[indt] = clock
            theta_t.append(copy(theta))
            epoch += Tmax/Nepochs

        for i in range(I):
            prob = theta[i].dot(p)
            o = np.random.choice(list(range(O)), p=prob)
            obs.append((i,o,indt))

        clock = Tmax*(nobs+1)/NobsperI
        theta = ftetnorm(clock)

    theta_t = np.array(theta_t)

    obs = np.array(obs, dtype=object)




    # for i in range(9):
    #     plt.subplot(3,3,i+1)
    #     plt.plot(theta_t[:, i, :])
    # plt.show()
    # sys.exit()

    return obs, theta_t, p, indt_to_time

def getDataRW(folder, ds):
    with open(f"{folder}/Data/{ds}_indt_to_time.pkl", "rb") as f:
        indt_to_time = pickle.load(f)
    with open(f"{folder}/Data/{ds}_observations.pkl", "rb") as f:
        obs = pickle.load(f)

    obs = np.array(obs, dtype=object)
    return obs, indt_to_time

def saveData(folder, codeSave, obs_train, obs_validation, obs_test, indt_to_time):
    curfol = "./"
    for fol in folder.split("/"):
        if fol not in os.listdir(curfol) and fol!="":
            os.mkdir(curfol+fol)
        curfol += fol+"/"

    with open(folder+codeSave+"obs_train.pkl", "wb+") as f:
        pickle.dump(obs_train, f)
    with open(folder+codeSave+"obs_validation.pkl", "wb+") as f:
        pickle.dump(obs_validation, f)
    with open(folder+codeSave+"obs_test.pkl", "wb+") as f:
        pickle.dump(obs_test, f)
    with open(folder+codeSave+"indt_to_time.pkl", "wb+") as f:
        pickle.dump(indt_to_time, f)

def saveParams(folder, codeSave, fitted_params):
    for fold in range(len(fitted_params)):
        theta_fin, p_fin, beta_fin = fitted_params[fold]
        np.save(folder+codeSave+f"f{fold}_theta.npy", theta_fin)
        np.save(folder+codeSave+f"f{fold}_p.npy", p_fin)
        np.save(folder+codeSave+f"f{fold}_beta.npy", beta_fin)

def saveTrueParams(folder, codeSave, theta, p):
    np.save(folder+codeSave+f"theta_true.npy", theta)
    np.save(folder+codeSave+f"p_true.npy", p)

def splitDS(obs, folds):
    allInds = list(range(len(obs)))
    np.random.shuffle(allInds)
    inds_test = np.array_split(allInds, folds)
    obs_train = []
    obs_validation = []
    obs_test = []
    for i in range(len(inds_test)):
        test, val = np.array_split(obs[inds_test[i]], 2)
        obs_test.append(test)
        obs_validation.append(val)
        inds_train = list(set(list(range(len(obs))))-set(inds_test[i]))
        obs_train.append(obs[inds_train])

    obs_train, obs_validation, obs_test = np.array(obs_train, dtype=object), np.array(obs_validation, dtype=object), np.array(obs_test, dtype=object)
    return obs_train, obs_validation, obs_test

def initVar(I,K,O,Nepochs):
    thetaPrev = []
    for indt in range(Nepochs):
        thetaPrev.append([])
        thetaPrev[indt] = np.random.random((I, K))
        thetaPrev[indt] /= thetaPrev[indt].sum(-1)[:, None]
    thetaPrev = np.array(thetaPrev)
    pPrev = np.random.random((K, O))
    pPrev /= pPrev.sum(-1)[:, None]
    return thetaPrev, pPrev

def log_prior(alpha_tr, thetaPrev, indt_to_time, beta, Nepochs, Nobs_epoch):
    vecPrior = []
    limNodes = 2

    listIndtToTime = np.array([indt_to_time[t] for t in range(Nepochs)])
    for indt in range(len(thetaPrev)):
        alphak, div = 0., 0.

        indsPrec = list(range(np.max((indt-limNodes, 0)), indt))
        Nobsprev = Nobs_epoch[indsPrec, None, None]
        timeDiffs = (listIndtToTime[indt]-listIndtToTime[indsPrec])[:, None, None]
        alphak += (thetaPrev[indsPrec]*Nobsprev/timeDiffs).sum(0)
        div += (Nobsprev/timeDiffs).sum()

        indsSuiv = list(range(indt+1, np.min((indt+1+limNodes, len(thetaPrev)))))
        Nobssuiv = Nobs_epoch[indsSuiv, None, None]
        timeDiffs = (listIndtToTime[indsSuiv]-listIndtToTime[indt])[:, None, None]
        alphak += (thetaPrev[indsSuiv]*Nobssuiv/timeDiffs).sum(0)
        div += (Nobssuiv/timeDiffs).sum()

        alphak /= div + 1e-20

        vecPrior.append(beta*alphak)  # alphak = 1 + beta*thetak

    return np.array(vecPrior)

def likelihood(alpha_tr, theta, p, indt_to_time, beta, Nepochs, Nobs_epoch):
    nnz = (alpha_tr>0).astype(bool)
    L = np.sum(np.log(alpha_tr[nnz] * (theta.dot(p))[nnz] + 1e-20))
    value_prior = 0

    if beta != 0:
        priors = log_prior(alpha_tr, theta, indt_to_time, beta, Nepochs, Nobs_epoch)
        value_prior = gammaln(np.sum(priors)) - np.sum(gammaln(priors)) + np.sum(priors*np.log(theta+1e-20))

    return L, L+value_prior

def maximizationTheta(obs, alphadivided, thetaPrev, p, indt_to_time, K, beta, alpha_tr, Nepochs, Nobs_epoch):
    theta = np.tensordot(alphadivided, p.T, axes=1)*thetaPrev

    vecPrior = log_prior(alpha_tr, thetaPrev, indt_to_time, beta, Nepochs, Nobs_epoch)

    theta += vecPrior

    theta /= theta.sum(axis=-1)[:, :, None]+1e-20

    # Equivalent but slower
    # phi = alpha_tr.sum(-1) + vecPrior.sum(-1)
    # theta /= phi[:, :, None] + 1e-20

    return theta

def maximizationP(obs, alphadivided, theta, pPrev, K, alpha_tr, Nobs_epoch):
    p = np.tensordot(theta, alphadivided, axes=((0,1), (0,1)))
    p = p*pPrev

    p /= p.sum(-1)[:, None]+1e-20

    return p

def evaluate(obs_test, theta, p, print_res=False):
    pred, true = [], []
    nbOut = p.shape[-1]

    for (i,o,indt) in obs_test:
        true_tmp = np.zeros((nbOut))
        true_tmp[o] = 1

        pred_tmp = theta[indt,i].dot(p)

        true.append(true_tmp)
        pred.append(pred_tmp)

    roc = roc_auc_score(true, pred, average="micro")
    ap = average_precision_score(true, pred, average="micro")
    F1 = f1_score(true, (np.array(pred)>0.5).astype(int), average="micro")
    if print_res:
        print("roc", "ap", "F1", sep="\t")
        print(roc, ap, F1, sep="\t")

    return [roc, ap, F1]

def run(obs_train, obs_validation, K, indt_to_time, nbLoops=1000, log_beta_bb=(-2, 3), res_beta=20, printProg=False, p_true=None, use_p_true=True, set_beta_null=False, one_epoch=False, rw=False):
    fitted_params = []

    I = 0
    O = 0
    Nepochs = 0
    for fold in range(len(obs_train)):
        if np.max(obs_train[fold][:, 0]) > I:
            I = np.max(obs_train[fold][:, 0])
        if np.max(obs_train[fold][:, 1]) > O:
            O = np.max(obs_train[fold][:, 1])
        if np.max(obs_train[fold][:, 2])>Nepochs:
            Nepochs = np.max(obs_train[fold][:, 2])
    I+=1
    O+=1
    Nepochs+=1

    beta_validation = np.append([0], np.logspace(log_beta_bb[0], log_beta_bb[1], res_beta))

    if one_epoch:
        Nepochs = 1
        for fold in range(len(obs_train)):
            for i, (item,o,indt) in enumerate(obs_train[fold]):
                obs_train[fold][i] = (item,o,0)
            for i, (item,o,indt) in enumerate(obs_validation[fold]):
                obs_validation[fold][i] = (item,o,0)

    if set_beta_null or one_epoch:
        beta_validation = [0]

    for fold in range(len(obs_train)):
        alpha_tr = np.zeros((Nepochs,I,O))
        for (i,o,indt) in obs_train[fold]:
            alpha_tr[indt,i,o] += 1

        Nobs_epoch = alpha_tr.sum(axis=(1,2))

        theta_init, p_init = initVar(I,K,O,Nepochs)
        theta_fin, p_fin, beta_fin, bestMetric = None, None, 0., -1e20

        tabRes, tabBeta = [], []

        for beta in beta_validation:
            thetaPrev, pPrev = copy(theta_init), copy(p_init)
            theta = thetaPrev
            p = pPrev

            Lprev = -1e20
            for iter_em in range(nbLoops):

                if iter_em%10==0:
                    L, L_prior = likelihood(alpha_tr, theta, p, indt_to_time, beta, Nepochs, Nobs_epoch)
                    if (Lprev-L)/Lprev<0.01 and rw:
                        print("BROKEN", (Lprev-L)/Lprev)
                        break
                    Lprev = L
                    print(f"{iter_em}/{nbLoops} - K={K} - B={beta} - L={L}")

                allProbs = np.tensordot(theta, p, axes=1)+1e-20
                alphadivided = alpha_tr/allProbs

                theta = maximizationTheta(obs_train[fold], alphadivided, thetaPrev, pPrev, indt_to_time, K, beta, alpha_tr, Nepochs, Nobs_epoch)

                if use_p_true:
                    assert p_true is not None
                    p = copy(p_true)
                else:
                    p = maximizationP(obs_train[fold], alphadivided, thetaPrev, pPrev, K, alpha_tr, Nobs_epoch)

                #L, L_prior = likelihood(alpha_tr, theta, p, indt_to_time, beta, Nepochs, Nobs_epoch)
                #if L==Lprev: break
                #print(f"{iter_em}/{nbLoops}", L, L_prior)
                #Lprev = L

                thetaPrev, pPrev = copy(theta), copy(p)

            roc = evaluate(obs_validation[fold], theta, p)[0]
            tabRes.append(roc)
            tabBeta.append(beta)
            if printProg:
                print(np.round(beta, 3), np.round(roc, 3), np.round(tabBeta[np.where(tabRes==np.max(tabRes))[0][0]], 3), np.round(np.max(tabRes), 3), sep="\t")

            if roc>bestMetric:
                theta_fin = copy(theta)
                p_fin = copy(p)
                beta_fin = beta
                bestMetric = roc

        fitted_params.append((theta_fin, p_fin, beta_fin))

    return fitted_params

# Varying NobsperI
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
        for NobsperI in np.linspace(Nepochs, Nepochs*100, 21):
            NobsperI = int(NobsperI)
            codeSave = f"{typeVar}_Nobs={NobsperI}_"
            print(codeSave)
            obs, theta_true, p_true, indt_to_time = getData(I,O,K,Nepochs, NobsperI, Tmax, typeVar=typeVar)
            obs_train, obs_validation, obs_test = splitDS(obs, folds)
            saveData(folder, codeSave, obs_train, obs_validation, obs_test, indt_to_time)
            saveTrueParams(folder, codeSave, theta_true, p_true)

            fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, log_beta_bb=(-2, 3), res_beta=res_beta, p_true=p_true, printProg=True)
            saveParams(folder, codeSave, fitted_params)
            fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, set_beta_null=True, p_true=p_true, printProg=False)
            saveParams(folder, codeSave+"beta_null_", fitted_params)
            fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, one_epoch=True, p_true=p_true, printProg=False)
            saveParams(folder, codeSave+"one_epoch_", fitted_params)

# Varying Nepochs
def XP2(folder = "XP/Synth/Nepochs/"):
    I = 100
    K = 3
    O = 3

    NobsperI = 1000
    Tmax = 2*np.pi
    nbLoops = 1000
    folds = 5
    res_beta = 40

    for typeVar in ["sin", "rnd"]:
        for Nepochs_div in reversed([1, 5, 10, 20, 30, 40, 50, 75, 100]):  # = Nobs moyen par epoque
            Nepochs = int(NobsperI/Nepochs_div)
            codeSave = f"{typeVar}_Nepochs={Nepochs}_"
            print(codeSave)
            obs, theta_true, p_true, indt_to_time = getData(I,O,K,Nepochs, NobsperI, Tmax, typeVar=typeVar)
            obs_train, obs_validation, obs_test = splitDS(obs, folds)
            saveData(folder, codeSave, obs_train, obs_validation, obs_test, indt_to_time)
            saveTrueParams(folder, codeSave, theta_true, p_true)

            fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, log_beta_bb=(-2, 3), res_beta=res_beta, p_true=p_true, printProg=False)
            saveParams(folder, codeSave, fitted_params)
            fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, set_beta_null=True, p_true=p_true, printProg=False)
            saveParams(folder, codeSave+"beta_null_", fitted_params)
            fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, one_epoch=True, p_true=p_true, printProg=False)
            saveParams(folder, codeSave+"one_epoch_", fitted_params)

# Varying p
def XP3(folder = "XP/Synth/VarP/"):
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
        for shiftp in np.linspace(0, 0.5, 21):
            obs, theta_true, p_true, indt_to_time = getData(I,O,K,Nepochs, NobsperI, Tmax, typeVar=typeVar, shiftp=shiftp)
            obs_train, obs_validation, obs_test = splitDS(obs, folds)
            for infer_p in [True, False]:
                codeSave = f"{typeVar}_shiftp={round(shiftp, 4)}_inferp={infer_p}_"

                print(codeSave)

                saveData(folder, codeSave, obs_train, obs_validation, obs_test, indt_to_time)
                saveTrueParams(folder, codeSave, theta_true, p_true)

                fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, log_beta_bb=(-2, 3), res_beta=res_beta, use_p_true=infer_p, p_true=p_true, printProg=False)
                saveParams(folder, codeSave, fitted_params)
                fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, set_beta_null=True, use_p_true=infer_p, p_true=p_true, printProg=False)
                saveParams(folder, codeSave+"beta_null_", fitted_params)
                fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, one_epoch=True, use_p_true=infer_p, p_true=p_true, printProg=False)
                saveParams(folder, codeSave+"one_epoch_", fitted_params)

# Real world XP
def XP4(folder="XP/RW/", ds="lastfm"):
    curfol = "./"
    for fol in (folder+ds+"/").split("/"):
        if fol not in os.listdir(curfol) and fol!="":
            os.mkdir(curfol+fol)
        curfol += fol+"/"

    folds = 5
    codeSave = ds+"_"
    nbLoops = 500
    log_beta_bb=(0, 5)
    if ds=="lastfm":  # Not the same time scale
        log_beta_bb=(2, 8)
    res_beta = 10

    obs, indt_to_time = getDataRW(folder, ds)
    obs_train, obs_validation, obs_test = splitDS(obs, folds)
    saveData(folder+f"{ds}/", codeSave, obs_train, obs_validation, obs_test, indt_to_time)

    for K in [5, 10, 20, 30]:
        tic = time.time()
        fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, log_beta_bb=log_beta_bb, res_beta=res_beta, use_p_true=False, printProg=True, rw=True)
        saveParams(folder+f"{ds}/", codeSave+f"{K}_", fitted_params)
        fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, set_beta_null=True, use_p_true=False, printProg=False, rw=True)
        saveParams(folder+f"{ds}/", codeSave+f"{K}_"+"beta_null_", fitted_params)
        fitted_params = run(obs_train, obs_validation, K, indt_to_time, nbLoops=nbLoops, one_epoch=True, use_p_true=False, printProg=False, rw=True)
        saveParams(folder+f"{ds}/", codeSave+f"{K}_"+"one_epoch_", fitted_params)
        print(f"K={K} - {np.round((time.time()-tic)/(3600), 2)}h elapsed =====================================")


XP = int(input("Which XP > "))
#XP = 4

if XP==1:
    XP1()
if XP==2:
    XP2()
if XP==3:
    XP3()
if XP==4:
    ds = str(input("Which DS > "))
    #ds = "lastfm"
    XP4(ds=ds)


# x,y = [], []
# for node in range(len(theta_true)):
#     x.append(indt_to_time[node])
#     y.append(theta_true[node][0])
# plt.plot(x,y,"k")
# x,y = [], []
# for node in range(len(theta_fin)):
#     x.append(indt_to_time[node])
#     y.append(theta_fin[node][0])
# plt.plot(x,y)
# plt.title(fr"$\beta={beta_fin}$")
# plt.show()
# print("=====")
# print()

#print(theta)

