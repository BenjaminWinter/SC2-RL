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
                    temp.append(float(obj['r']))
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
    plt.plot(mean, 'g', label="Mean", alpha=0.3)

    smoothed = smooth(mean, 5000, 'hanning')
    #fig.add_subplot(2, cols, fnum+1)
    plt.plot(smoothed, 'g', label="smooth")

    plt.show()


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

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
