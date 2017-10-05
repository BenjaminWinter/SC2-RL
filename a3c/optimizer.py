import multiprocessing as mp
import a3c.shared as shared


class Optimizer(mp.Process):
    stop_signal = False

    def __init__(self, thread_num, brain=None):
        super(Optimizer, self).__init__()
        self.thread_num = thread_num
        self.brain = brain

    def run(self):
        while not self.stop_signal:
            self.brain.optimize()

    def stop(self):
        self.stop_signal = True

