import numpy as np
import pandas as pd
import pickle

file = "reddit.csv"
altscale=True

for file in ["lastfm.csv", "wikipedia.csv", "reddit.csv", "mooc.csv"]:
    print(file)
    timeslice = 0.
    if "reddit" in file:
        timeslice = 24*3600  # 1j
        if altscale: timeslice = 24*3600/24  # 1h
    if "wikipedia" in file:
        timeslice = 24*3600  # 1j
        if altscale: timeslice = 24*3600/24  # 1h
    if "lastfm" in file:
        timeslice = 24*3600*3  # 3j
        if altscale: timeslice = 24*3600  # 1j

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

    alttxt = ""
    if altscale: alttxt="_alt"
    with open("XP/RW/Data/"+file.replace('.csv', alttxt+'_indt_to_time.pkl'), "wb+") as f:
        pickle.dump(indt_to_time, f)
    with open("XP/RW/Data/"+file.replace('.csv', alttxt+'_observations.pkl'), "wb+") as f:
        pickle.dump(obs, f)

