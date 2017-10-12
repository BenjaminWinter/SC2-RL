import threading
import a3c.shared as shared


class Optimizer(threading.Thread):

    def __init__(self, thread_num):
        super(Optimizer, self).__init__()
        self.thread_num = thread_num
        self.stop_value = False

    def run(self):
        while not self.stop_value:
            shared.brain.optimize()

    def stop(self):
        self.stop_value = True