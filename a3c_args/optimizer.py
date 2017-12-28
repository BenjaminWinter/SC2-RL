import threading

class Optimizer(threading.Thread):

    def __init__(self, thread_num, brain=None, stop=None):
        super(Optimizer, self).__init__()
        self.thread_num = thread_num
        self.brain = brain
        self.stop = stop

    def run(self):
        while not self.stop.value:
            self.brain.optimize()