import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def __init__(self, emb_dim=50, hidden_layer_size=600, data_shape=(28,28)):
        super(VAE, self).__init__()

        self.emb_dim = emb_dim
        self.hidden_layer_size = hidden_layer_size
        self.data_shape = data_shape

        self.enc_fc_1 = nn.Linear(self.data_shape[0] * self.data_shape[1], self.hidden_layer_size)
        self.enc_fc_2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.enc_fc_3_mu = nn.Linear(self.hidden_layer_size, self.emb_dim)
        self.enc_fc_3_sigma = nn.Linear(self.hidden_layer_size, self.emb_dim)

        self.dec_fc_1 = nn.Linear(self.emb_dim, self.hidden_layer_size)
        self.dec_fc_2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.dec_fc_3 = nn.Linear(self.hidden_layer_size, self.data_shape[0] * self.data_shape[1])

    def encode(self, x):
        x = torch.flatten(x,1)
        x = self.enc_fc_1(x)
        x = F.softplus(x)
        x = self.enc_fc_2(x)
        x = F.softplus(x)
        mu = F.softplus(self.enc_fc_3_mu(x))
        sigma = F.softplus(self.enc_fc_3_sigma(x))
        return mu, sigma

    def decode(self, z):
        z = self.dec_fc_1(z)
        z = F.softplus(z)
        z = self.dec_fc_2(z)
        z = F.softplus(z)
        z = self.dec_fc_3(z)
        x_hat = torch.sigmoid(z)
        x_hat = torch.reshape(x_hat, (-1, self.data_shape[0], self.data_shape[1]))
        return x_hat

    def forward(self, x):
        mu, sigma = self.encode(x)
        eps = torch.randn(mu.shape[1]).to(device)
        z = mu + sigma*eps
        x_hat = self.decode(z)
        return x_hat, mu, sigma

    @staticmethod
    def vae_loss(x, mu, sigma, x_hat):
        decoder_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        encoder_loss = - 0.5 * torch.sum(1 + 2 * torch.log(sigma) - mu ** 2 - sigma ** 2)
        return decoder_loss + encoder_loss, decoder_loss, encoder_loss

