from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn
from .renderer import NeRFRenderer

#TODO: create a new class NeRFWNetwork by copying NeRFNetwork
class NeRFWNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 cuda_ray=False,
                 in_channels_a=16,
                 in_channels_t=16,
                 N_vocab = 100
                 ):
        super().__init__(bound, cuda_ray)

        #TODO: add appearance (and later transient) embeddings similar to https://github.com/kwea123/nerf_pl/blob/nerfw/train.py#L42
        # add l_a l_t to the model

        self.embedding_a = torch.nn.Embedding(N_vocab, in_channels_a)
        self.embedding_t = torch.nn.Embedding(N_vocab, in_channels_t)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.in_channels_a = in_channels_a 
        self.in_channels_t = in_channels_t

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )


        self.in_dim_color_s = self.encoder_dir.n_output_dims + self.geo_feat_dim + self.in_channels_a

        #TODO: replace color_net by 2 networks: static_net and transient_net with appropriate input and output dim
        self.color_net_s = tcnn.Network(
            n_input_dims=self.in_dim_color_s,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

        self.in_dim_color_t = self.geo_feat_dim + self.in_channels_t

        # self.color_net_t = tcnn.Network(
        #     n_input_dims=self.in_dim_color_t,
        #     n_output_dims=3 + 1 + 1,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": hidden_dim_color,
        #         "n_hidden_layers": num_layers_color - 1,
        #     },
        # )
        self.color_net_t_sigma = tcnn.Network(
            n_input_dims=self.in_dim_color_t,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "Softplus",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

        self.color_net_t_rgb = tcnn.Network(
            n_input_dims=self.in_dim_color_t,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "Sigmoid",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

        self.color_net_t_beta = tcnn.Network(
            n_input_dims=self.in_dim_color_t,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "Softplus",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )
    
    def forward(self, x, d, l_a, l_t, only_static = False):
        
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]
        # l_a: [B, N, in_channels_a=16]
        # l_t: [B, N, in_channels_t =16]

        #TODO: add additional inputs, app_emb and trans_emb, with default set to None
        
        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        d = d.view(-1, 3)
        l_a = l_a.view(-1,self.in_channels_a)

        # sigma_s
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        sigma_s = F.relu(h[..., 0]).unsqueeze(-1)
        geo_feat = h[..., 1:]

        # color_s
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        # p = torch.zeros_like(geo_feat[..., :1]) # manual input padding 
        h_s = torch.cat([d, geo_feat, l_a], dim=-1)
        h_s = self.color_net_s(h_s)
        
        # sigmoid activation for rgb
        color_s = torch.sigmoid(h_s)
    
        sigma_s = sigma_s.view(*prefix, -1)
        color_s = color_s.view(*prefix, -1)
        static = torch.cat([color_s, sigma_s], dim=-1)

        # # transient sigma and color
        # h_t = torch.cat([geo_feat, l_t], dim=-1)
        # h_t = self.color_net_t(h_t)
        # # problem avec torch.split?
        # color_t = h_t[..., :3]
        # sigma_t = h_t[..., 3].unsqueeze(-1)
        # beta = h_t[..., 4].unsqueeze(-1)

        # color_t = torch.sigmoid(color_t)
        # sigma_t = torch.nn.softplus(sigma_t)
        # beta  = torch.nn.softplus(beta)

        # static = torch.cat([color_s, sigma_s], dim=-1)
        # transient = torch.cat([color_t, sigma_t, beta], dim=-1)
        
        if not only_static: 
            # transient sigma and color
            l_t = l_t.view(-1,self.in_channels_t)
            h_t = torch.cat([geo_feat, l_t], dim=-1)
            sigma_t = self.color_net_t_sigma(h_t)
            color_t = self.color_net_t_rgb(h_t)
            beta = self.color_net_t_beta(h_t)

            # problem avec torch.split?
            # color_t = h_t[..., :3]
            # sigma_t = h_t[..., 3].unsqueeze(-1)
            # beta = h_t[..., 4].unsqueeze(-1)

            # color_t = torch.sigmoid(color_t)
            # sigma_t = torch.nn.Softplus(sigma_t)
            # beta  = torch.nn.Softplus(beta)

            color_t = color_t.view(*prefix, -1)
            sigma_t = sigma_t.view(*prefix, -1)
            beta = beta.view(*prefix, -1)


            transient = torch.cat([color_t, sigma_t, beta], dim=-1)

            return static, transient
        else:
            return static

    def density(self, x):
        # x: [B, N, 3], in [-bound, bound]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        #sigma = tor

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 cuda_ray=False,
                 ):
        super().__init__(bound, cuda_ray)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        #TODO: add appearance (and later transient) embeddings similar to https://github.com/kwea123/nerf_pl/blob/nerfw/train.py#L42


        self.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )


        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        #TODO: replace color_net by 2 networks: static_net (and later transient_net) with appropriate input and output dim
        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    
    def forward(self, x, d):
        
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]
        #TODO: add additional inputs, app_emb and trans_emb, with default set to None
        
        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        d = d.view(-1, 3)

        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
    
        sigma = sigma.view(*prefix)
        color = color.view(*prefix, -1)

        return sigma, color

    def density(self, x):
        # x: [B, N, 3], in [-bound, bound]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        #sigma = tor