import numpy as np

class DriftDetector:

    def __init__(self):
        self.history = []

    def update(self, error):
        self.history.append(error)
        if len(self.history) > 200:
            self.history.pop(0)

    def check_drift(self):
        if len(self.history) < 50:
            return False

        recent = np.mean(self.history[-50:])
        past = np.mean(self.history[:50])

        return recent > past * 1.5