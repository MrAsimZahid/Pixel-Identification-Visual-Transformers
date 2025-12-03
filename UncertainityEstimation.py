import torch
import torch.nn as nn

class BayesianViTRegressor(nn.Module):
    """ViT with uncertainty estimation using Monte Carlo Dropout"""
    def __init__(self, base_model, n_samples=10):
        super().__init__()
        self.base_model = base_model
        self.n_samples = n_samples
        
        # Enable dropout at inference
        self.enable_dropout()
    
    def enable_dropout(self):
        """Enable dropout layers for uncertainty estimation"""
        for m in self.base_model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()  # Keep dropout active
    
    def forward(self, x, n_samples=None):
        n_samples = n_samples or self.n_samples
        
        predictions = []
        for _ in range(n_samples):
            pred = self.base_model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [n_samples, batch, 2]
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Uncertainty (standard deviation)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty