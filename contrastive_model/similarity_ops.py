import torch
import torch.nn as nn
import torch.nn.functional as F
from contrastive_model.complex_stuff import CplxLinear


class COCOLASimilarity(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(data=torch.Tensor(self.dim, self.dim))
        self.w.data.normal_(0, 0.05)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = F.tanh(self.layer_norm(x))
        y = F.tanh(self.layer_norm(y))
        return torch.matmul(x, torch.matmul(self.w, y.t()))

    def pairwise(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = F.tanh(self.layer_norm(x))
        y = F.tanh(self.layer_norm(y))
        return (x * torch.matmul(self.w, y.t()).t()).sum(dim=-1)
    
class BilinearSimilarity(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.projection = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.projection.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.projection(F.normalize(x, p=2, dim=-1))
        y = F.normalize(y, p=2, dim=-1)
                
        return x @ y.t()

    def pairwise(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.projection(F.normalize(x, p=2, dim=-1))
        y = F.normalize(y, p=2, dim=-1)

        return (x * y).sum(dim=-1)

def complex_normalize(x: torch.Tensor) -> torch.Tensor:
    norm = ((x ** 2).sum(dim=(-2,-1), keepdim=True) + 1e-6) ** .5
    return x / norm


class CplxCosineSimilarity(nn.Module):
    def __init__(self, dim, temp_init=0.07) -> None:
        super().__init__()
        self.dim = dim
        self.projection = CplxLinear(dim, dim, bias=False)
        with torch.no_grad():
            self.projection.W.weight.fill_(0)
            self.projection.W.weight[:dim, :] = torch.eye(dim)

        # Learnable log-temperature for stability
        self.logit_scale = nn.Parameter(torch.tensor(torch.log(torch.tensor(1 / temp_init))))

    def _complex_inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the magnitude of the Hermitian inner product |x^H y|.
        x and y are expected to be shape (..., 2, D)
        """
        # x_re: (..., D), x_im: (..., D)
        x_re, x_im = x[..., 0, :], x[..., 1, :]
        y_re, y_im = y[..., 0, :], y[..., 1, :]

        # Real part of x^H y: (x_re * y_re + x_im * y_im)
        # Imag part of x^H y: (x_re * y_im - x_im * y_re)
        # Using einsum for flexible batch/matrix multiplication
        
        # All-to-all dot products (Matrix Multiplication)
        if x.dim() == 3 and y.dim() == 3: # Input: (B, 2, D)
            real_inner = torch.matmul(x_re, y_re.t()) + torch.matmul(x_im, y_im.t())
            imag_inner = torch.matmul(x_re, y_im.t()) - torch.matmul(x_im, y_re.t())
        # Element-wise dot products (Pairwise)
        else:
            real_inner = (x_re * y_re + x_im * y_im).sum(-1)
            imag_inner = (x_re * y_im - x_im * y_re).sum(-1)

        # Magnitude squared: |z|^2 = re^2 + im^2
        # We return the magnitude (sqrt) for cosine similarity
        return torch.sqrt(real_inner**2 + imag_inner**2 + 1e-9)

    def forward(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Normalize and project
        x = complex_normalize(self.projection(complex_normalize(x))) # B, 2, D
        y = complex_normalize(self.projection(complex_normalize(y))) # B, 2, D
        
        # Result shape: (B, B) representing all-to-all similarity
        return self._complex_inner_product(x, y) * torch.clamp(self.logit_scale.exp(), max=100.0)
    
    def pairwise(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Normalize and project
        x = complex_normalize(self.projection(complex_normalize(x))) # B, 2, D
        y = complex_normalize(self.projection(complex_normalize(y))) # B, 2, D
        
        # Result shape: (B,) representing 1-to-1 similarity
        return self._complex_inner_product(x.unsqueeze(1), y.unsqueeze(1)).squeeze() * torch.clamp(self.logit_scale.exp(), max=100.0)


class CplxBilinearSimilarity(nn.Module):
    def __init__(self, dim, psd=False, hermitian=False) -> None:
        super().__init__()
        self.dim = dim
        assert not (psd and hermitian), "Cannot be both PSD and Hermitian."
        self.psd = psd
        self.hermitian = hermitian

        self.W = CplxLinear(dim, dim, bias=False)
        
        # Initialize weights to match COCOLA's scale (0.05)
        # with torch.no_grad():
        #     self.W.W.weight.normal_(0, 0.05)

    def forward(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes Re(x^H W y) for all pairs in the batch.
        Output: (B, B) real-valued logits.
        """
        # 1. Normalize
        x = complex_normalize(x) # B, 2, D
        y = complex_normalize(y) # B, 2, D
        
        # 2. Project X using the complex weight matrix W
        proj_x = self.W(x) # B, 2, D
        if self.psd or self.hermitian:
            proj_y = self.W(y)
            if self.psd:
                y = proj_y

        term = torch.matmul(proj_x[..., 0, :], y[..., 0, :].t())
        term += torch.matmul(proj_x[..., 1, :], y[..., 1, :].t())
        if self.hermitian:
            term += torch.matmul(x[..., 0, :], proj_y[..., 0, :].t())
            term += torch.matmul(x[..., 1, :], proj_y[..., 1, :].t())
            term /= 2.0
        
        return term

    def pairwise(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Computes Re(x^H W y) for matched pairs.
        Output: (B,) real-valued logits.
        """
        x = complex_normalize(x)
        y = complex_normalize(y)
        
        x = self.W(x) # B, 2, D
        if self.psd:
            y = self.W(y)
        
        x_re, x_im = x[..., 0, :], x[..., 1, :]
        y_re, y_im = y[..., 0, :], y[..., 1, :]
        
        # Element-wise dot product
        term1 = (x_re * y_re).sum(dim=-1)
        term2 = (x_im * y_im).sum(dim=-1)
        
        return term1 + term2