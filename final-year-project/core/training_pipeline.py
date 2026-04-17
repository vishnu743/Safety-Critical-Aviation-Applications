import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from .adaptive_model import AdaptiveAutoencoder
from .data_stream import SensorStream
from .drift_detector import DriftDetector
from .uncertainty_monitor import UncertaintyMonitor
from intelligence.counterfactual_engine import CounterfactualEngine
from intelligence.explainability import ExplainabilityEngine


SENSORS = ["sensor_2","sensor_3","sensor_4","sensor_7","sensor_11"]


class AdaptiveTrainingPipeline:

    def __init__(self, dataset_path):

        # -------- Data stream ----------
        self.stream = SensorStream(dataset_path, SENSORS)

        # -------- Monitoring modules ----------
        self.drift = DriftDetector()
        self.uncertainty_monitor = UncertaintyMonitor()

        self.retrain_dataset = []

        # -------- Load dataset for initial training ----------
        # load and normalize dataset
        df = pd.read_excel(dataset_path)
        # make column names lowercase for consistency
        df.columns = df.columns.str.lower()

        # some datasets use 'time_cycle' instead of 'cycle'
        if "time_cycle" in df.columns and "cycle" not in df.columns:
            df = df.rename(columns={"time_cycle": "cycle"})

        # sensors list may be provided in lowercase; ensure we match lowercase columns
        sensor_cols = [s.lower() for s in SENSORS]

        healthy = df[df["cycle"] <= 50][sensor_cols].values

        # -------- Model ----------
        self.model = AdaptiveAutoencoder(len(SENSORS))
        self.model.initial_train(healthy)

        # -------- Counterfactual Engine (initialized after model) ----------
        self.counterfactual = CounterfactualEngine(self.model)

        # -------- Explainability ----------
        self.explainer = ExplainabilityEngine(
            self.model.model,
            self.model.scaler,
            SENSORS
        )

        print("Initial training complete")


    # =====================================================
    # REAL TIME EXECUTION LOOP
    # =====================================================
    def run(self, steps=100):

        for _ in range(steps):

            sample = self.stream.next()

            error, uncertainty = self.model.predict_with_uncertainty(sample)

            print("Error:", round(error,5),
                "Uncertainty:", round(uncertainty,5))

            self.uncertainty_monitor.update(uncertainty)

            if error < 0.01:
                self.retrain_dataset.append(sample)

            if error > 0.03:
                explanation = self.explainer.explain(sample)
                print("🚨 CRITICAL CONDITION")
                print("Top contributors:", explanation["top_contributors"])

            if error > 0.02:  # WARNING or higher
                cf_results = self.counterfactual.generate(sample)
                boundaries = self.counterfactual.find_failure_boundary(sample)

                print("\n🔍 Counterfactual Analysis:")
                print(cf_results[:3])  # show few results

                print("\n⚠️ Failure Boundaries:")
                print(boundaries)

            if self.uncertainty_monitor.should_retrain():
                if len(self.retrain_dataset) > 200:
                    print("⚙ Automatic retraining triggered")
                    retrain_data = np.array(self.retrain_dataset)
                    self.model.retrain_full(retrain_data)
                    self.retrain_dataset = []

            time.sleep(1)

        print("Simulation finished")