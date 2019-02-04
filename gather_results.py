"""Script for plotting tensorflow summaries from multiple models.

Plot description is stored on separate .json file
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
import sys
import json


WINDOW = 10 ** 6
STEP_LIMIT = 25 * 10 ** 6
TAG = 'Perf/RewardTotal'

with open(sys.argv[1]) as config_file:
    plot_config = json.load(config_file)

plt.figure(figsize=(8, 4))

for plot_name, logs in plot_config.items():
    print('Plotting', plot_name)

    plot_filename = plot_name + '.pdf'

    # Check modification date to avoid unnecessary replotting
    if os.path.exists(plot_filename):
        plot_time = os.path.getmtime(plot_filename)

        skip = True
        for log in logs:
            for log_path in log[1:]:
                if os.path.exists(log_path):
                    log_time = os.path.getmtime(log_path)
                    if log_time > plot_time:
                        skip = False
                        break
            if not skip:
                break

        if skip:
            print('No changes. Skipping.')
            continue

    data = []

    for log in logs:
        log_name = log[0]
        log_paths = log[1:]

        print('Gathering', log_name, 'in', log_paths)

        values = []

        for path in log_paths:
            for dir in os.walk(path):
                for filename in dir[2]:
                    if 'tfevents' not in filename:
                        continue
                    print('Reading', filename)

                    record = tf.train.summary_iterator(
                        os.path.join(dir[0], filename))

                    try:
                        for event in record:
                            if event.step > STEP_LIMIT:
                                break

                            for value in event.summary.value:
                                if value.tag == TAG:
                                    values.append(
                                        (event.step, value.simple_value))
                    except tf.errors.DataLossError:
                        print('Error, skipping', filename)

        values.sort()

        if len(values) == 0:
            print('No values logged for', log_name)
            values.append((0, 0))

        t, y = zip(*values)
        mean = []
        stdev = []

        i = 0
        for j in range(len(values)):
            while t[j] - t[i] > WINDOW:
                i += 1

            mean.append(np.mean(y[i:j + 1]))
            # stdev.append(np.std(y[i:j + 1]))
            stdev.append(scipy.stats.sem(y[i:j + 1]))

        data.append(
            (log_name,
                {'t': np.array(t),
                    'mean': np.array(mean),
                    'stdev': np.array(stdev),
                }
            )
        )

    print('Drawing')

    plt.clf()

    for log_name, trace in data:
        # plt.fill_between(
        #     data[log]['t'],
        #     data[log]['mean'] - data[log]['stdev'],
        #     data[log]['mean'] + data[log]['stdev'],
        #     alpha=0.2
        # )
        plt.plot(trace['t'], trace['mean'], label=log_name)

    plt.legend()
    plt.xlim(0, STEP_LIMIT)
    plt.grid()
    plt.xlabel('Actor actions')
    plt.ylabel('Average reward per episode')
    plt.savefig(plot_filename, bbox_inches='tight', format='pdf')
