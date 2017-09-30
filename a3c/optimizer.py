import threading
import shared as shared


class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self, thread_num):
        threading.Thread.__init__(self)
        self.thread_num = thread_num

    def run(self):
        while not self.stop_signal:
            shared.brain.optimize()

    def stop(self):
        self.stop_signal = True

