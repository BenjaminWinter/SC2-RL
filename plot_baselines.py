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
    cols = math.ceil((total + 6) // 3)
    fig = plt.figure()
    #plt.ylim(0,100)
    for fname in glob.glob(os.path.join(args.dir, '*.json')):
        with open(fname) as file:
            line = file.readline()
            temp = []
            while line:
                obj = json.loads(line)
                if 'r' in obj:
                    temp.append(float(obj['r']))
                line = file.readline()
            rewards.append(temp)
            ax = fig.add_subplot(3,cols,fnum+1)
            print(fnum+1)
            plt.plot(temp, 'c',label="Plot" + str(fnum), alpha=0.3)

            smoothed = smooth(np.array(temp), 5000, 'hanning')
            plt.plot(smoothed, 'c', label="smoothed")
        fnum += 1
    minsize = min([len(x) for x in rewards])
    print(minsize)
    rewards = np.array([x[:minsize] for x in rewards])
    highest = np.amax(rewards)

    max = np.amax(rewards, 0)
    mean = np.mean(rewards, 0)
    mean_last = mean[-1000:]

    print('Highest Overall Score: {}'.format(highest))
    print('Average Score over last 1000 Episodes: {}'.format(np.average(mean_last)))

    fig.add_subplot(3, cols, fnum+1)
    plt.ylim(-1, 50)
    plt.xlim(0, 103000)
    plt.plot(max, 'r', label="Max", alpha=0.3)
    smoothed = smooth(max, 3000, 'flat')
    plt.plot(smoothed, 'r', label="Max")
    fnum += 1

    fig.add_subplot(3, cols, fnum+1)
    plt.ylim(-1, 50)
    plt.xlim(0, 103000)
    plt.plot(mean, 'orange', label="Mean", alpha=0.3)

    smoothed = smooth(mean, 3000, 'flat')
    plt.plot(smoothed, 'orange', label="smooth")

    plt.show()


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError( "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError( "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


if __name__ == '__main__':
    main()
