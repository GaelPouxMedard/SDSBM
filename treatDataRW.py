import numpy as np
import pandas as pd
import pickle

file = "reddit.csv"

for file in ["reddit.csv", "lastfm.csv", "wikipedia.csv", "mooc.csv"]:
    print(file)
    timeslice = 0.
    if "reddit" in file:
        timeslice = 24*3600
    if "wikipedia" in file:
        timeslice = 24*3600
    if "lastfm" in file:
        timeslice = 24*3600*30

    indt_to_time = {}
    indt = 0
    endPeriod = timeslice*(indt+1)  # 0
    obs = []
    timeprec = -1e20
    with open("XP/RW/Data/"+file.replace('.csv', '_observations.txt'), "w+") as o:
        with open("XP/RW/"+file, "r") as f:
            labels = f.readline().split(",")
            for line in f:
                line_splitted = line.replace("\n", "").split(",")
                usr, itm, time = line_splitted[:3]
                time=float(time)

                if time>endPeriod:
                    indt += 1
                    endPeriod = timeslice*(indt+1)
                    indt_to_time[indt] = timeslice*indt  # We keep the starting point

                o.write(f"{usr}\t{itm}\t{indt}\n")

                if timeprec>time:  # Check that data is sorted
                    pause()
                timeprec=time



    with open("XP/RW/Data/"+file.replace('.csv', 'indt_to_time.pkl'), "wb+") as f:
        pickle.dump(indt_to_time, f)

