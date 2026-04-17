import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib


class AdaptiveAutoencoder:

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.scaler = MinMaxScaler()
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(16, activation='relu')(input_layer)
        encoded = Dense(8, activation='relu')(encoded)
        decoded = Dense(16, activation='relu')(encoded)
        output_layer = Dense(self.input_dim, activation='linear')(decoded)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        return model

    def initial_train(self, data):
        scaled = self.scaler.fit_transform(data)
        self.model.fit(scaled, scaled, epochs=10, batch_size=32, verbose=0)
        self.save()

    def predict_with_uncertainty(self, sample):
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        scaled = self.scaler.transform(sample)
        reconstruction = self.model.predict(scaled, verbose=0)
        error = np.mean((scaled - reconstruction) ** 2)
        uncertainty = np.std((scaled - reconstruction) ** 2)
        return error, uncertainty

    def retrain_full(self, data):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        scaled = self.scaler.fit_transform(data)
        self.model.fit(scaled, scaled, epochs=5, batch_size=32, verbose=0)
        self.save()

    def save(self):
        """Save model and scaler to files"""
        try:
            # Try to save as newer .keras format first
            self.model.save("models/model.keras")
        except Exception as e:
            print(f"Warning: Could not save as .keras format: {e}")
            # Fallback to .h5 format
            try:
                self.model.save("models/model.h5")
            except Exception as e2:
                print(f"Error: Could not save model: {e2}")
        
        joblib.dump(self.scaler, "models/scaler.pkl")

    def load(self):
        """Load model and scaler from files"""
        import os
        
        # Try loading newer .keras format first
        if os.path.exists("models/model.keras"):
            try:
                self.model = load_model("models/model.keras")
                self.scaler = joblib.load("models/scaler.pkl")
                print("Successfully loaded .keras model")
                return
            except Exception as e:
                print(f"Could not load .keras model: {e}")
        
        # Fallback to .h5 format (with custom_objects to handle deserialization issues)
        if os.path.exists("models/model.h5"):
            try:
                import tensorflow as tf
                # Try loading with custom_objects empty to avoid deserialization errors
                custom_objs = {}
                self.model = load_model("models/model.h5", custom_objects=custom_objs)
                self.scaler = joblib.load("models/scaler.pkl")
                print("Successfully loaded .h5 model")
                return
            except Exception as e:
                print(f"Could not load .h5 model ({e}) - creating new model instead")
                # Create new model if loading fails
                self.model = self.build_model()
                return
        
        # If no model files exist, create a fresh one
        print("No existing model found - creating new model")
        self.model = self.build_model()