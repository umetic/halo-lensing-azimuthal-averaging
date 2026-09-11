"""NFW accuracy, exhaustive branch handling, physical normalization and integral tests."""
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize import brentq

from azlens.cosmology import PUBLICATION_COSMOLOGY as COSMO
from azlens.nfw import nfw_f,nfw_g,nfw_shear_shape,nfw_lensing,nfw_mass_shape,nfw_normalization,nfw_density,nfw_surface_density

REF=Path(__file__).resolve().parents[1]/'reference/foundation'
HIST=json.loads((REF/'historical_targets.json').read_text())
HIGH=json.loads((REF/'nfw_high_precision.json').read_text())['rows']


@pytest.mark.parametrize('func,key',[(nfw_f,'f'),(nfw_g,'G'),(nfw_shear_shape,'h')])
def test_high_precision_reference(func,key):
    x=np.array([float.fromhex(row['x_hex']) for row in HIGH])
    values=func(x)
    assert np.all(np.isfinite(values))
    assert_allclose(values,[row[key] for row in HIGH],rtol=2e-12,atol=0)


@pytest.mark.parametrize('edge',[.001,.999,1.001,.99999,1.00001,.999999,1.000001])
def test_threshold_and_adjacent_floats(edge):
    x=np.array([np.nextafter(edge,0),edge,np.nextafter(edge,np.inf)])
    for fn in (nfw_f,nfw_g,nfw_shear_shape):
        values=fn(x)
        assert np.all(np.isfinite(values)) and np.all(values>0)
        assert_allclose(values,np.full(3,values[1]),rtol=2e-12,atol=0)


def test_exact_unity_limits_and_small_radius():
    assert_allclose(nfw_f(1),1/3,rtol=0,atol=1e-16)
    assert_allclose(nfw_g(1),1-np.log(2),rtol=0,atol=1e-16)
    assert_allclose(nfw_shear_shape(1),10/3-4*np.log(2),rtol=2e-15)
    assert_allclose(nfw_shear_shape([1e-12,1e-10]),1,rtol=1e-15)


def test_legacy_f_away_from_edge_cases():
    row=HIST['legacy_surface_shape']
    assert_allclose(nfw_f(row['x']),row['f'],rtol=2e-12,atol=0)


def independent_f(x):
    # LOS integral of rho/(rho_s)=1/[r(1+r)^2], after r=x*sec(theta).
    return quad(lambda theta:np.cos(theta)/(x+np.cos(theta))**2,0,np.pi/2,epsabs=2e-13,epsrel=2e-13,limit=200)[0]


@pytest.mark.parametrize('x',[.01,.1,.5,1.,2.,10.,100.])
def test_surface_shape_against_los_quadrature(x):
    assert_allclose(nfw_f(x),independent_f(x),rtol=3e-12,atol=0)


@pytest.mark.parametrize('x',[.2,1.,3.])
def test_enclosed_profile_independent_double_integral(x):
    expected=quad(lambda y:y*independent_f(y),0,x,epsabs=3e-12,epsrel=3e-12,limit=150)[0]
    assert_allclose(nfw_g(x),expected,rtol=2e-11,atol=1e-13)


def test_g_derivative_identity():
    x=np.array([.02,.1,.7,1.,1.8,10.])
    step=2e-4*x
    # Five-point derivative; the x=1 stencil exercises the near-unity series.
    derivative=(nfw_g(x-2*step)-8*nfw_g(x-step)+8*nfw_g(x+step)-nfw_g(x+2*step))/(12*step)
    assert_allclose(derivative,x*nfw_f(x),rtol=2e-10,atol=0)


def test_lensing_broadcast_and_normalization():
    x=np.array([1e-10,.2,1.,3.])[None,:]
    amp=np.array([0.,.1,.3])[:,None]
    fields=nfw_lensing(x,amp)
    assert fields.kappa.shape==(3,4)
    assert_allclose(fields.mean_kappa-fields.kappa,fields.gamma_t,rtol=2e-14,atol=1e-15)
    assert_allclose(fields.kappa[2],3*fields.kappa[1],rtol=2e-15)
    assert_allclose(fields.gamma_t[0],0,rtol=0,atol=0)
    assert nfw_f(1).shape==()
    assert nfw_f(np.zeros((0,2))+1).shape==(0,2)


@pytest.mark.parametrize('row',HIST['normalization'])
def test_historical_normalizations(row):
    out=nfw_normalization(row['M'],row['c'],row['z_l'])
    for name,key in [('r200c_mpc_h','r200c'),('r_s_mpc_h','r_s'),('rho_s_hunits','rho_s'),('kappa_s_infinity','kappa_s_infinity')]:
        assert_allclose(getattr(out,name),row[key],rtol=5e-13,atol=0)


def test_normalization_arrays_and_3d_mass_integral():
    out=nfw_normalization(np.array([3e14,1e15,2e15]),[3.,3.843,6.],.3)
    assert out.r_s_mpc_h.shape==(3,)
    scalar=nfw_normalization(1e15,3.843,.3)
    r200=float(scalar.r200c_mpc_h)
    # Integrate a dimensionless fraction of the actual density rather than using f_M again.
    fraction=quad(lambda u:4*np.pi*(u*r200)**2*float(nfw_density(u*r200,scalar))*r200/1e15,0,1,epsabs=2e-13,epsrel=2e-13)[0]
    assert_allclose(fraction,1,rtol=3e-13)
    radii=scalar.r_s_mpc_h*np.array([.1,1.,3.])
    sigma=nfw_surface_density(radii,scalar)
    fields=nfw_lensing(radii/scalar.r_s_mpc_h,scalar.kappa_s_infinity)
    assert_allclose(sigma/COSMO.sigma_crit_infinity_hunits(.3),fields.kappa,rtol=3e-15)


def test_appendix_d_centered_safety_radius_anchor():
    out=nfw_normalization(1e15,3.843,.3)
    # No ring solver: for a centered sphere, lambda_minus=1-mean_kappa.
    root=brentq(lambda r:1-float(nfw_lensing(r*3.843,out.kappa_s_infinity).mean_kappa)-.1,.001,.5,xtol=1e-14)
    assert_allclose(root,0.053811316516287774,rtol=3e-11,atol=0)


@pytest.mark.parametrize('bad',[0.,-1.,np.nan,np.inf])
def test_invalid_profile_inputs(bad):
    for fn in (nfw_f,nfw_g,nfw_shear_shape,nfw_mass_shape):
        with pytest.raises(ValueError): fn(bad)
    with pytest.raises(ValueError): nfw_normalization(bad,4,.3)
    with pytest.raises(ValueError): nfw_normalization(1e15,bad,.3)


def test_invalid_amplitude_and_lens():
    with pytest.raises(ValueError): nfw_lensing(1,-1)
    with pytest.raises(ValueError): nfw_lensing(1,np.nan)
    with pytest.raises(ValueError): nfw_normalization(1e15,4,0)


def test_small_c_mass_shape():
    c=np.array([1e-10,1e-8])
    assert_allclose(nfw_mass_shape(c)/c**2,.5-2*c/3,rtol=2e-15)
