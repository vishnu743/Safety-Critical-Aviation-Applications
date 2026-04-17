import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.training_pipeline import AdaptiveTrainingPipeline

pipeline = AdaptiveTrainingPipeline("data/train_001_final.xlsx")
pipeline.run(steps=60)