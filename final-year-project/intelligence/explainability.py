import numpy as np

class ExplainabilityEngine:

    def __init__(self, model, scaler, sensor_names):
        self.model = model
        self.scaler = scaler
        self.sensor_names = sensor_names

    def explain(self, sample):
        scaled = self.scaler.transform(sample.reshape(1,-1))
        reconstruction = self.model.predict(scaled, verbose=0)

        # Per-sensor reconstruction error
        per_feature_error = np.abs(scaled - reconstruction)[0]

        # Rank sensors by contribution
        sorted_indices = np.argsort(per_feature_error)[::-1]

        explanation = []

        for idx in sorted_indices:
            explanation.append({
                "sensor": self.sensor_names[idx],
                "contribution": float(per_feature_error[idx])
            })

        total_error = float(np.mean(per_feature_error))

        return {
            "total_error": total_error,
            "top_contributors": explanation[:3]
        }