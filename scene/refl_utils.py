import torch
import numpy as np
import math


def get_ml_array(deg_view):
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))
    ml_array = np.array(ml_list).T
    return ml_array



def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return (np.sqrt(
        (2.0 * l + 1.0) * np.math.factorial(l - m) /
        (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))

def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
      l: associated Legendre polynomial degree.
      m: associated Legendre polynomial order.
      k: power of cos(theta).

    Returns:
      A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
            np.math.factorial(l - k - m) *
            generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))



class generate_ide_fn(torch.nn.Module):
    def __init__(self, deg_view):
        super().__init__()

        if deg_view > 5:
            print('WARNING: Only deg_view of at most 5 is numerically stable.')

        self.ml_array = get_ml_array(deg_view)
        l_max = 2**(deg_view - 1)

        self.mat = torch.zeros((l_max + 1, self.ml_array.shape[1])).cuda()
        for i, (m, l) in enumerate(self.ml_array.T):
            for k in range(l - m + 1):
                self.mat[k, i] = sph_harm_coeff(l, m, k)

    
    def integrated_dir_enc_fn(self,xyz, kappa_inv):

        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]
        vmz = torch.cat([z**i for i in range(self.mat.shape[0])], axis=-1)
        vmxy = torch.cat([(x + 1j * y)**m for m in self.ml_array[0, :]], axis=-1)
        sph_harms = vmxy * torch.matmul(vmz, self.mat)
        sigma = 0.5 * self.ml_array[1, :] * (self.ml_array[1, :] + 1)
        ide = sph_harms * torch.exp(-torch.tensor(sigma, device='cuda') * kappa_inv)
        return torch.cat([torch.real(ide), torch.imag(ide)], axis=-1).float()