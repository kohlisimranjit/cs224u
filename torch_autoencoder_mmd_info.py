import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase
from utils import progress_bar
from torch_autoencoder import TorchAutoencoder
from torch.autograd import Variable

__author__ = "Simranjit"
__version__ = "CS224u, Stanford, Spring 2020"

class MMDVAE(torch.nn.Module):
    def __init__(self, input_dim_, hidden_dim, output_dim_, hidden_activation):
        super(MMDVAE, self).__init__()
        self.hidden_activation = hidden_activation
        self.input_dim_ = input_dim_
        self.hidden_dim = hidden_dim
        self.output_dim_ = output_dim_

        self.linear1 = nn.Linear(self.input_dim_, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim_)

    def forward(self, x):
        l1=self.linear1(x)
        z = self.hidden_activation(l1)
        x_hat = self.linear2 (z)
        return x_hat, z
    
class TorchAutoencoderMMDIG(TorchAutoencoder):
   
    def __init__(self, **kwargs):
        super(TorchAutoencoderMMDIG, self).__init__(**kwargs)
        

    def define_graph(self):
        return MMDVAE(self.input_dim_, self.hidden_dim, self.output_dim_, self.hidden_activation)

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd

    def sample_and_compute_mmd(self, z):
        true_samples = Variable(
                    torch.randn(z.shape),
                    requires_grad=False
                ).to(self.device)
        mmd = self.compute_mmd(true_samples, z)
        return mmd

    def fit(self, X):
        # Data prep:
        self.input_dim_ = X.shape[1]
        self.output_dim_ = X.shape[1]
        # Dataset:
        X_tensor = self.convert_input_to_tensor(X)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=True)
        # Graph
        if not self.warm_start or not hasattr(self, "model"):
            self.model = self.define_graph()
            self.opt = self.optimizer(
                self.model.parameters(),
                lr=self.eta,
                weight_decay=self.l2_strength)
        self.model.to(self.device)
        self.model.train()
        # Optimization:
        loss = nn.MSELoss()
        # Train:
        for iteration in range(1, self.max_iter+1):
            epoch_error = 0.0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                batch_preds, z = self.model(X_batch)
                recons_err = loss(batch_preds, y_batch) 
                print("recons "+ str(recons_err))
                mmd_err = self.sample_and_compute_mmd(z)
                print("mmd_err "+ str(mmd_err))
                epoch_error += recons_err.item()+mmd_err.item()
                print("epoch_error "+str(epoch_error))
                err = recons_err + mmd_err
                self.opt.zero_grad()
                err.backward()
                self.opt.step()
            self.errors.append(epoch_error)
            progress_bar(
                "Finished epoch {} of {}; error is {}".format(
                    iteration, self.max_iter, err))
        # Hidden representations:
        with torch.no_grad():
            self.model.to('cpu')
            X_hat, H = self.model(X_tensor)
            return self.convert_output(H, X)


    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = self.convert_input_to_tensor(X)
            self.model.to('cpu')
            X_pred , Z= self.model(X_tensor)
            return self.convert_output(X_pred, X)

    @staticmethod
    def convert_output(X_pred, X):
        X_pred = X_pred.cpu().numpy()
        if isinstance(X, pd.DataFrame):
            X_pred = pd.DataFrame(X_pred, index=X.index)
        return X_pred


def simple_example():
    import numpy as np

    np.random.seed(seed=42)

    def randmatrix(m, n, sigma=0.1, mu=0):
        return sigma * np.random.randn(m, n) + mu

    rank = 20
    nrow = 1000
    ncol = 100

    X = randmatrix(nrow, rank).dot(randmatrix(rank, ncol))
    ae = TorchAutoencoderMMDIG(hidden_dim=rank, max_iter=2)
    H = ae.fit(X)
    X_pred = ae.predict(X)
    mse = (0.5*(X_pred - X)**2).mean()
    print("\nMSE between actual and reconstructed: {0:0.06f}".format(mse))
    print("Hidden representations")
    print(H)
    return mse


if __name__ == '__main__':
   simple_example()
