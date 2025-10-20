import torch
from torch.nn import Module, Parameter
from torch.nn import init
import torch.nn.functional as F

class GCNLayer(Module):  # Renamed from GraphConvolution
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        self.linear_transform = Parameter(torch.Tensor(input_dim, output_dim))
        
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()  

    def reset_parameters(self):
        init.kaiming_uniform_(self.linear_transform)  
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, original_features, augmented_features, adjacency_matrix):
     
        
        transformed_original = torch.mm(original_features, self.linear_transform)
        output_original = torch.spmm(adjacency_matrix, transformed_original)
        
        transformed_augmented = torch.mm(augmented_features, self.linear_transform)
        output_augmented = torch.spmm(adjacency_matrix, transformed_augmented)

        if self.use_bias:
            output_original += self.bias
            output_augmented += self.bias
        
        output_original = F.relu(output_original)
        output_augmented = F.relu(output_augmented)
        
        return output_original, output_augmented

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'
               
  
class GDNLayer(Module):  
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GDNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
    
        self.linear_transform = Parameter(torch.Tensor(input_dim, output_dim))
        
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()  # Initialize parameters

    def reset_parameters(self):
        init.kaiming_uniform_(self.linear_transform)  
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, original_features, augmented_features, adjacency_matrix):
    
        
        transformed_original = torch.mm(original_features, self.linear_transform)
        output_original = torch.spmm(adjacency_matrix, transformed_original)
        
        transformed_augmented = torch.mm(augmented_features, self.linear_transform)
        output_augmented = torch.spmm(adjacency_matrix, transformed_augmented)

        if self.use_bias:
            output_original += self.bias
            output_augmented += self.bias
        
        output_original = F.relu(output_original)
        output_augmented = F.relu(output_augmented)
        
        return output_original, output_augmented

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.input_dim} -> {self.output_dim})"       
    
    
class GCAE(Module):
     def __init__(self, input_dim, latent_dim, output_dim):
        super(GCAE, self).__init__()
        # Updated layer names and consistent variable naming
        self.encoder = GCNLayer(input_dim, latent_dim)    # Was z_layer (GraphConvolution)
        self.decoder = GDNLayer(latent_dim, output_dim)  # Was x_hat_layer (GraphDeconvolution)

     def forward(self, original_features, augmented_features, adjacency_matrix):
        """Encoder (GCN)"""
        latent_original, latent_augmented = self.encoder(
            original_features=original_features,
            augmented_features=augmented_features,
            adjacency_matrix=adjacency_matrix
        )
        
        """Decoder (GDN)"""
        reconstructed_original, reconstructed_augmented = self.decoder(
            original_features=latent_original,
            augmented_features=latent_augmented,
            adjacency_matrix=adjacency_matrix
        )

        return latent_original, latent_augmented, reconstructed_original, reconstructed_augmented      