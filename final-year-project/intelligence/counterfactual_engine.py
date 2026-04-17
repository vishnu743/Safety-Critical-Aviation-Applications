import numpy as np

class CounterfactualEngine:
    def __init__(self, model):
        self.model = model

    def generate(self, sample, steps=5, change=0.05):
        """
        Generate counterfactual scenarios
        steps: number of variations
        change: % change per step
        """
        sample = np.array(sample)
        results = []

        for i in range(len(sample)):  # each sensor
            for direction in [-1, 1]:  # decrease & increase
                modified = sample.copy()

                for step in range(1, steps + 1):
                    factor = 1 + (direction * change * step)
                    modified[i] = sample[i] * factor

                    error, _ = self.model.predict_with_uncertainty(modified)

                    results.append({
                        "sensor": f"sensor_{i}",
                        "step": step,
                        "change_%": direction * change * step * 100,
                        "error": float(error),
                        "status": self.get_status(error)
                    })

                    # Stop early if critical found
                    if error > 0.03:
                        break

        return results

    def get_status(self, error):
        if error < 0.01:
            return "NORMAL"
        elif error < 0.03:
            return "WARNING"
        return "CRITICAL"

    def find_failure_boundary(self, sample):
        """
        Find the exact threshold where each sensor triggers failure (error > 0.03)
        
        Returns list of sensors with their failure thresholds
        """
        boundaries = []

        for i in range(len(sample)):
            original = sample[i]

            for change in np.linspace(-0.3, 0.3, 20):  # -30% to +30% in 20 steps
                modified = sample.copy()
                modified[i] = original * (1 + change)

                error, _ = self.model.predict_with_uncertainty(modified)

                if error > 0.03:
                    boundaries.append({
                        "sensor": f"sensor_{i}",
                        "threshold_change_%": change * 100,
                        "trigger_error": float(error)
                    })
                    break

        return boundaries