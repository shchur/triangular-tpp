from datetime import datetime
import numpy as np


def datetimes_to_dataset(times, dst_file):
    days = [[times[0]]]
    current_day = days[0]
    for t in times[1:]:
        if t.date() != current_day[0].date():
            current_day = []
            days.append(current_day)
        # Skipping duplicates
        if len(current_day) > 0 and t == current_day[-1]:
            continue
        current_day.append(t)
    for i in range(len(days)):
        date = datetime.combine(days[i][0].date(), datetime.min.time())
        days[i] = np.sort(np.array([(t - date).total_seconds()/3600 for t in days[i]]))
    mean_number_items = np.mean([len(day) for day in days])
    max_time = 24 # seconds in a day
    np.savez(dst_file, arrival_times=days, nll=np.zeros(len(days)), t_max=max_time, mean_number_items=mean_number_items)
