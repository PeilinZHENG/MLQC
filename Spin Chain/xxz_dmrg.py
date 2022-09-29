import numpy as np
import scipy
import matplotlib.pyplot as plt 
import tenpy
import tenpy.linalg.np_conserved as npc
import os
import warnings
warnings.filterwarnings("ignore")
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

from tenpy.networks.mps import build_initial_state

from tenpy.models.lattice import Site, Chain
from tenpy.models.model import CouplingModel, NearestNeighborModel, MPOModel, CouplingMPOModel
from tenpy.linalg import np_conserved as npc
from tenpy.tools.params import asConfig
from tenpy.networks.site import SpinHalfSite  # if you want to use the predefined site

class XXZChain2(CouplingMPOModel, NearestNeighborModel):
    r"""Another implementation of the Spin-1/2 XXZ chain with Sz conservation.

    This implementation takes the same parameters as the :class:`XXZChain`, but is implemented
    based on the :class:`~tenpy.models.model.CouplingMPOModel`.

    Parameters
    ----------
    model_params : dict | :class:`~tenpy.tools.params.Config`
        See :cfg:config:`XXZChain`
    """
    def __init__(self, model_params):
        model_params = asConfig(model_params, "XXZChain2")
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        return SpinHalfSite(conserve=model_params.get('Conserve', None))  # use predefined Site

    def init_terms(self, model_params):
        # read out parameters
        Jxx = model_params.get('Jxx', 1.)
        Jz = model_params.get('Jz', 1.)
        hz = model_params.get('hz', 0.)
        # add terms
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hz, u, 'Sz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(Jxx * 0.5, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)

DefaultPath = '/home/sijing/ML/MPS/'


def mydmrg(Jxx):
    model_params = {
    'Jxx': Jxx , 'hz': 0, 'Jz': -1,
    # 'L': 5,
    'bc':'periodic',
    'bc_MPS': 'infinite',
    # 'bc_x': 'open',
    'conserve': None,
    }

    M = XXZChain2(model_params)

    # psi = MPS.from_lat_product_state(M.lat, [['up']])
    L = 2
    chi = 8
    # site = SpinHalfSite(conserve=None)  # predefined charges and Sp,Sm,Sz operators
    # lat = Chain(L, site, bc_MPS='infinite', bc='periodic')
    lat = M.lat
    p_state = []
    
    def random_SpinHalfMPS_translationally_different(L, chi, func=np.random.rand, bc='infinite', form='B'):
        d = 2
        Bs = []
        for i in range(L):
            B = func( d, chi, chi )*2-1+ 1j*(func( d, chi, chi )*2-1)
            Bs.append(B)
        psi = MPS.from_Bflat(lat.mps_sites(), Bs, bc=lat.bc_MPS, dtype=None)
        psi.canonical_form(renormalize=True)
        return psi


    psi = random_SpinHalfMPS_translationally_different(L=L, chi=chi)

    dmrg_params = {
    'mixer': None,  # setting this to True helps to escape local minima
    'max_E_err': 1.e-5,
    # 'max_sweeps': 100,
    'N_sweeps_check':10,
    'norm_tol': 1.e-3,
    'max_S_err':3.e-3,
    'trunc_params': {
        'chi_max': 100,
        'svd_min': 1.e-10
    },
    'verbose': True,
    'combine': True
    }

    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()
    return E, psi




if __name__ == "__main__":

    L = 6

    opt = 'DMRG'

    if not os.path.exists(DefaultPath + 'OUT/mps_app/L={}/{}'.format(L, opt)):
        os.mkdir(DefaultPath + 'OUT/mps_app/L={}/{}'.format(L, opt))
    if not os.path.exists(DefaultPath + 'OUT/mps_app/L={}/{}/xxz'.format(L, opt)):
        os.mkdir(DefaultPath + 'OUT/mps_app/L={}/{}/xxz'.format(L, opt))


    dmrg_E = []

    op = []

    Jxx = []

    for i in range(75):
        print('Jxx=', -0.5 - 0.02*i)

        E, psi = mydmrg(Jxx = -0.5 - 0.02*i)
        Jxx.append(-0.5 - 0.02*i)
        # print('Energy per site:', E)
        dmrg_E.append(E)

        op1 = []
        op1.append(psi.expectation_value_term([('Sx', 0)]))
        op1.append(psi.expectation_value_term([('Sy', 0)]))
        op1.append(psi.expectation_value_term([('Sz', 0)]))
        op1.append(psi.expectation_value_term([('Sz', 0), ('Sz', 5)]))
        op1 = np.asarray(op1)

        op.append(op1)

        np.save(DefaultPath + 'OUT/mps_app/L={}/{}/xxz/E.npy'.format(L, opt), dmrg_E)
        np.save(DefaultPath + 'OUT/mps_app/L={}/{}/xxz/Jxx.npy'.format(L, opt), Jxx)
        np.save(DefaultPath + 'OUT/mps_app/L={}/{}/xxz/op.npy'.format(L, opt), op)

        print('Jxx={} Done!'.format(-0.5 - 0.02*i))
    
    np.save(DefaultPath + 'OUT/mps_app/L={}/{}/xxz/E.npy'.format(L, opt), dmrg_E)
    np.save(DefaultPath + 'OUT/mps_app/L={}/{}/xxz/Jxx.npy'.format(L, opt), Jxx)
    np.save(DefaultPath + 'OUT/mps_app/L={}/{}/xxz/op.npy'.format(L, opt), op)

    

