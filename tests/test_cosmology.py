"""Publication normalization checks; no Colossus population is constructed."""
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from azlens.cosmology import PublicationCosmology, PUBLICATION_COSMOLOGY as COSMO, C_LIGHT_KMS

TARGETS = json.loads((Path(__file__).resolve().parents[1]/"reference/foundation/historical_targets.json").read_text())


def test_asymptotic_references():
    assert_allclose(COSMO.omega_r, TARGETS['cosmology']['omega_r'], rtol=3e-15)
    assert_allclose(COSMO.comoving_distance_infinity_mpc_h(), TARGETS['cosmology']['chi_infinity'], rtol=3e-13)


@pytest.mark.parametrize('row', TARGETS['cosmology']['rows'])
def test_historical_source_scaling(row):
    z = row['z_l']
    assert_allclose(COSMO.comoving_distance_mpc_h(z), row['chi_l'], rtol=3e-13)
    assert_allclose(COSMO.beta_infinity(z), row['beta_infinity'], rtol=3e-13)
    assert_allclose(COSMO.sigma_crit_infinity_hunits(z), row['sigma_crit_infinity'], rtol=4e-13)
    for source in row['weights']:
        zs = source['z_s']
        assert_allclose(COSMO.beta(z,zs),source['beta'],rtol=3e-13)
        assert_allclose(COSMO.source_weight(z,zs),source['w'],rtol=3e-13)
        assert_allclose(COSMO.sigma_crit_hunits(z,zs)*source['w'],row['sigma_crit_infinity'],rtol=4e-13)
        assert 0 < COSMO.beta(z,zs) < COSMO.beta_infinity(z) < 1


@pytest.mark.parametrize('z',[.2,.3,.5,1.,2.,5.])
def test_independent_redshift_distance_integral(z):
    # Change integration variable to z, rather than reusing the scale-factor integrand.
    om, orad = COSMO.omega_m, COSMO.omega_r
    integrand = lambda zz: 1/np.sqrt(orad*(1+zz)**4 + om*(1+zz)**3 + 1-om-orad)
    expected=C_LIGHT_KMS/100*quad(integrand,0,z,epsabs=2e-13,epsrel=2e-13)[0]
    assert_allclose(COSMO.comoving_distance_mpc_h(z),expected,rtol=4e-13)


def test_infinite_distance_independent_variable():
    om, orad = COSMO.omega_m, COSMO.omega_r
    expected=C_LIGHT_KMS/100*quad(lambda z:1/np.sqrt(orad*(1+z)**4+om*(1+z)**3+1-om-orad),0,np.inf,epsabs=2e-12,epsrel=2e-12,limit=300)[0]
    assert_allclose(COSMO.comoving_distance_infinity_mpc_h(),expected,rtol=3e-12)


def test_two_backgrounds_are_distinct():
    z=np.array([0.,.2,.3,.5,1.])
    assert_allclose(COSMO.e2_distance(z)-COSMO.e2_late(z),COSMO.omega_r*((1+z)**4-1),rtol=3e-11,atol=3e-16)
    assert COSMO.comoving_distance_mpc_h(0)==0
    assert COSMO.beta(0,1)==1


def test_density_mass_definition_broadcast():
    m=np.array([3e14,1e15,2e15])[:,None]
    z=np.array([.2,.5])[None,:]
    r=COSMO.r200c_mpc_h(m,z)
    assert r.shape==(3,2)
    assert_allclose(4*np.pi/3*200*COSMO.rho_crit_hunits(z)*r**3,np.broadcast_to(m,r.shape),rtol=3e-15)


@pytest.mark.parametrize('z',[-1,np.nan,np.inf])
def test_invalid_redshifts(z):
    with pytest.raises(ValueError): COSMO.comoving_distance_mpc_h(z)
    with pytest.raises(ValueError): COSMO.e2_late(z)
    with pytest.raises(ValueError): COSMO.beta_infinity(z)


@pytest.mark.parametrize('zl,zs',[(.3,.3),(.5,.3),(-1,1),(.3,np.inf),(.3,np.nan)])
def test_invalid_source_order(zl,zs):
    with pytest.raises(ValueError): COSMO.source_weight(zl,zs)


def test_invalid_inputs_are_not_silently_clipped():
    with pytest.raises(ValueError): COSMO.sigma_crit_infinity_hunits(0)
    with pytest.raises(ValueError): COSMO.r200c_mpc_h([-1,1e15],.3)
    with pytest.raises(ValueError): COSMO.comoving_distance_mpc_h([.2,.3])
    with pytest.raises(ValueError): PublicationCosmology(h=0)
    with pytest.raises(ValueError): PublicationCosmology(omega_m=.03,omega_b=.049)
