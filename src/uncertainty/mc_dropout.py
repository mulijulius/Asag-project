import numpy as np
import torch

class MCDropout:
    def __init__(self, model, dropout_rate=0.5):
        self.model = model
        self.dropout_rate = dropout_rate
        self.model.train()  # Set the model to training mode to enable dropout

    def run_mc_dropout(self, embeddings, n_passes=100):
        all_passes = []

        for _ in range(n_passes):
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = self.model(embeddings)
                all_passes.append(outputs.numpy())

        all_passes = np.array(all_passes)
        mean_prediction = np.mean(all_passes, axis=0)
        std_prediction = np.std(all_passes, axis=0)
        confidence_scores = 1 - std_prediction / (std_prediction + 1e-8)  # Avoid division by zero

        return mean_prediction, std_prediction, all_passes, confidence_scores
