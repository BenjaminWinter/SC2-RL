import threading
import a3c.shared as shared


class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            shared.brain.optimize()

    def stop(self):
        self.stop_signal = True

