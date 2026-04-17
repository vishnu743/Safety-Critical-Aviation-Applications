import numpy as np

class UncertaintyMonitor:

    def __init__(self, window_size=50, threshold=0.015):
        self.window_size = window_size 
        self.threshold = threshold
        self.history = []
        
    def compute_uncertainty(self, reconstruction_errors):
        uncertainty = reconstruction_errors.std()
        self.history.append(uncertainty)
        return uncertainty

    def check_drift(self, reconstruction_errors):
        uncertainty = self.compute_uncertainty(reconstruction_errors)
        return uncertainty > self.threshold

    def update(self, uncertainty):
        # make sure window_size is initialized (defensive in case __init__ was skipped)
        if not hasattr(self, "window_size") or self.window_size is None:
            # fall back to default
            self.window_size = 50
        self.history.append(uncertainty)

        if len(self.history) > self.window_size:
            self.history.pop(0)

    def should_retrain(self):
        if len(self.history) < self.window_size:
            return False

        avg_uncertainty = np.mean(self.history)

        if avg_uncertainty > self.threshold:
            print("⚠ Sustained high uncertainty detected")
            return True

        return False