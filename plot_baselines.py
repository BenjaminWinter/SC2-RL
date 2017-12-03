from matplotlib import pyplot as plt
import json, argparse, glob, os, math
import numpy as np
import scipy.signal as signal

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', help='Log Directory', default='logs/results')
    args = parser.parse_args()
    rewards = []
    fnum = 0
    total = len(glob.glob(os.path.join(args.dir, '*.json')))
    cols = math.ceil((total + 2) // 2)
    fig = plt.figure()
    for fname in glob.glob(os.path.join(args.dir, '*.json')):
        with open(fname) as file:
            line = file.readline()
            temp = []
            while line:
                obj = json.loads(line)
                if 'r' in obj:
                    temp.append(obj['r'])
                line = file.readline()
            rewards.append(temp)
            ax = fig.add_subplot(2,cols,fnum+1)
            print(fnum+1)
            plt.plot(temp, label="Plot" + str(fnum))
        fnum += 1
    minsize = min([len(x) for x in rewards])
    print(minsize)
    rewards = np.array([x[:minsize] for x in rewards])

    max = np.amax(rewards, 0)
    mean = np.mean(rewards, 0)
    fig.add_subplot(2, cols, fnum+1)
    plt.plot(max, 'r', label="Max")
    fnum += 1
    fig.add_subplot(2, cols, fnum+1)
    plt.plot(mean, 'g', label="Mean")
    plt.show()

if __name__ == '__main__':
    main()
