import os
import numpy as np
import matplotlib.pyplot as plt
import logging


def adjust_learning_rate(optimizer, epoch, learning_rate):
    # lr_adjust = {epoch: learning_rate * (0.2 ** (epoch // 2))}
    lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logging.info(f"Updating learning rate to {lr}")


plt.switch_backend('agg')


def visual(true, preds=None, name='./test_pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def save_to_txt(root_dir):
    npy_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('metrics.npy'):
                npy_files.append(os.path.join(root, file))

    mode = 'a' if os.path.exists('results.txt') else 'w'
    with open('results.txt', mode) as f:
        for npy_file in npy_files:
            folder_name = os.path.basename(os.path.dirname(npy_file))

            data = np.load(npy_file, allow_pickle=True)
            data_str = str(data)
            f.write(f'The results from {npy_file}:\n{data_str}\n\n')

    logging.info(f"Best results have been written to results.txt")
