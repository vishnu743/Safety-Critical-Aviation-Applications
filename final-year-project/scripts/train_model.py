import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.training_pipeline import AdaptiveTrainingPipeline

if __name__ == "__main__":
    print("🚀 Training model...")

    pipeline = AdaptiveTrainingPipeline("data/train_001_final.xlsx")

    print("✅ Model training completed")