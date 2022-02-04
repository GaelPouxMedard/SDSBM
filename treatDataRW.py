import pickle
import sys
import numpy as np


def redoEpigraphy():
    file = "ClaussSlaby.csv"

    listToReplace = [("0", ""),
                     ("diplomata militaria", ""),
                     ("signacula medicorum", ""),
                     ("inscriptiones christianae", ""),
                     ("litterae erasae", ""),
                     ("litterae in litura", ""),
                     ("senatus consulta", ""),
                     ("sigilla impressa", ""),
                     ("tesserae nummulariae", ""),
                     ("tituli fabricationis", ""),
                     ("tituli honorarii", ""),
                     ("tituli operum", ""),
                     ("tituli possessionis", ""),
                     ("tituli sacri", ""),
                     ("tituli sepulcrales", ""),
                     ("nomen singulare", "nomen_singulare"),
                     ("ordo decurionum", "ordo_decurionum"),
                     ("ordo equester", "ordo_equester"),
                     ("ordo senatorius", "ordo_senatorius"),
                     ("praenomen et nomen", "praenomen_et_nomen"),
                     ("sacerdotes christiani", "sacerdotes_christiani"),
                     ("sacerdotes pagani", "sacerdotes_pagani"),
                     ("seviri augustales", "seviri_augustales"),
                     ("tria nomina", "tria_nomina"),
                     ("augusti augustae", "augusti_augustae"),
                     ("servi servae", "servi_servae"),
                     ("officium professio", "officium_professio"),
                     ("liberti libertae", "liberti_libertae"),
                     ("carmina", ""),
                     ("defixiones", ""),
                     ("leges", ""),
                     ("miliaria", ""),
                     ("signacula", ""),
                     ("termini", ""),
                     ]

    obs = []
    with open("XP/RW/Data/"+file, "r") as f:
        f.readline()
        for line in f:
            infos = line.split("\t")
            date = infos[1]
            date = date.replace("[", "").replace("]", "").split(", ")
            if float(date[1])-float(date[0])>50: continue
            date = (float(date[1])+float(date[0]))/2
            region = infos[2]
            if region=="province": continue
            titres = infos[5].lower()
            for orig, repl in listToReplace:
                titres = titres.replace(orig.lower(), repl.lower())
            titres = titres.split()

            for titre in titres:
                obs.append((titre, region, date))

    obs = np.array(obs, dtype=object)
    obs = [(i, o, t) for t, i, o in sorted(zip(obs[:, 2], obs[:, 0], obs[:, 1]))]

    ind_to_title, ind_to_region = {}, {}
    title_to_ind, region_to_ind = {}, {}
    itit, ireg = 0, 0
    txt = "user_id,item_id,timestamp,\n"
    tmin = -100
    for i, o, t in obs:
        if i not in title_to_ind:
            title_to_ind[i] = itit
            ind_to_title[itit] = i
            itit += 1
        if o not in region_to_ind:
            region_to_ind[o] = ireg
            ind_to_region[ireg] = o
            ireg += 1
        if t<tmin or t>500: continue
        t -= tmin
        t = t*365.25*24*3600
        txt += f"{title_to_ind[i]},{region_to_ind[o]},{t},\n"

    print(tmin)
    with open("XP/RW/Data/epigraphy.csv", "w+") as f:
        f.write(txt)
    with open("XP/RW/Data/epigraphy_indsToTitles.pkl", "wb+") as f:
        pickle.dump(ind_to_title, f)
    with open("XP/RW/Data/epigraphy_indsToRegions.pkl", "wb+") as f:
        pickle.dump(ind_to_region, f)
    with open("XP/RW/Data/epigraphy_tmin.pkl", "wb+") as f:
        pickle.dump(tmin, f)

#redoEpigraphy()

altscale=True

for file in ["reddit.csv", "epigraphy.csv", "lastfm.csv", "wikipedia.csv", "mooc.csv"]:
    print(file)
    timeslice = 0.
    if "reddit" in file:
        timeslice = 24*3600  # 1j
        if altscale: timeslice = 24*3600/2  # 12h
    if "wikipedia" in file:
        timeslice = 24*3600  # 1j
        if altscale: timeslice = 24*3600/24  # 1h
    if "lastfm" in file:
        timeslice = 24*3600*15  # 3j
        if altscale: timeslice = 24*3600*3  # 1j
    if "epigraphy" in file:
        timeslice = 24*3600*365*1  # 1y
        if altscale: timeslice = 24*3600*365*5  # 5y

    indt_to_time = {}
    indt = 0
    endPeriod = timeslice*(indt+1)
    indt_to_time[indt] = 0.
    obs = []
    timeprec = -1e20

    with open("XP/RW/Data/"+file, "r") as f:
        labels = f.readline().split(",")
        for line in f:
            line_splitted = line.replace("\n", "").split(",")
            usr, itm, time = line_splitted[:3]
            time=float(time)
            usr = int(usr)
            itm = int(itm)

            if time>endPeriod:
                indt += 1
                endPeriod = timeslice*(indt+1)
                indt_to_time[indt] = timeslice*indt  # We keep the starting point

            obs.append((usr, itm, indt))

            if timeprec>time:  # Check that data is sorted
                pause()
            timeprec=time

    obs2 = np.array(obs, dtype=object)
    print(np.unique(obs2[:, 2], return_counts=True))

    alttxt = ""
    if altscale: alttxt="_alt"
    with open("XP/RW/Data/"+file.replace('.csv', alttxt+'_indt_to_time.pkl'), "wb+") as f:
        pickle.dump(indt_to_time, f)
    with open("XP/RW/Data/"+file.replace('.csv', alttxt+'_observations.pkl'), "wb+") as f:
        pickle.dump(obs, f)

