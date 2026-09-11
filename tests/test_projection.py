"""Geometric identities, historical comparisons, and independent LOS integrals."""
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from azlens.projection import (
    volume_preserving_stretches, intrinsic_shape_matrix, sky_basis_from_los,
    project_triaxial, project_triaxial_batch, projected_radius,
    area_preserving_radius,
)
from azlens.nfw import nfw_f

ROWS=json.loads((Path(__file__).resolve().parents[1]/'reference/foundation/historical_targets.json').read_text())['projection']


@pytest.mark.parametrize('row',ROWS)
def test_scalar_historical_and_invariants(row):
    p,s,los=row['p'],row['s'],row['los']
    out=project_triaxial(p,s,los)
    a,b,c=volume_preserving_stretches(p,s)
    assert_allclose(a*b*c,1,rtol=2e-15)
    assert_allclose(np.linalg.det(out.shape_matrix),1,rtol=3e-15)
    assert_allclose(out.projected_matrix,row['P'],rtol=2e-13,atol=3e-15)
    assert_allclose([out.q_perp,out.b_los,out.scale_factor],[row['q_perp'],row['b_los'],row['scale_factor']],rtol=3e-13)
    assert_allclose(np.linalg.det(out.projected_matrix),out.b_los**2,rtol=6e-14)
    assert_allclose(out.eigenvalues,[out.b_los*out.q_perp,out.b_los/out.q_perp],rtol=6e-14)
    assert 0 < out.q_perp <= 1
    assert_allclose(out.sky_basis.T@out.sky_basis,np.eye(2),rtol=0,atol=5e-16)
    assert_allclose(out.sky_basis.T@out.los,0,rtol=0,atol=5e-16)
    assert_allclose(np.cross(out.sky_basis[:,0],out.sky_basis[:,1]),out.los,rtol=0,atol=5e-16)


def test_batch_matches_scalar_and_legacy():
    batch=project_triaxial_batch([r['p'] for r in ROWS],[r['s'] for r in ROWS],[r['los'] for r in ROWS])
    for name in ('q_perp','b_los','scale_factor'):
        assert_allclose(getattr(batch,name),[r['batch'][name] for r in ROWS],rtol=5e-13,atol=3e-15)
        assert_allclose(getattr(batch,name),[getattr(project_triaxial(r['p'],r['s'],r['los']),name) for r in ROWS],rtol=5e-13,atol=3e-15)
    repeated=project_triaxial_batch(.8,.5,[[1,0,0],[0,1,0],[0,0,1]])
    assert_allclose(repeated.q_perp,[.5/.8,.5,.8],rtol=5e-14)


@pytest.mark.parametrize('los',[[1,0,0],[0,1,0],[0,0,1],[1,2,3],[-2,1,.2]])
def test_spherical_limit(los):
    out=project_triaxial(1,1,los)
    assert_allclose(out.projected_matrix,np.eye(2),rtol=0,atol=8e-16)
    assert_allclose([out.q_perp,out.b_los,out.scale_factor],1,rtol=2e-15)
    assert_allclose(projected_radius([[1,0],[3,4]],out),[1,5],rtol=2e-15)


@pytest.mark.parametrize('axis',[0,1,2])
def test_principal_axis_projections(axis):
    p,s=.8,.5
    out=project_triaxial(p,s,np.eye(3)[axis])
    stretches=volume_preserving_stretches(p,s)
    assert_allclose(out.b_los,stretches[axis],rtol=3e-15)
    assert_allclose(out.q_perp,[s/p,s,p][axis],rtol=3e-15)


@pytest.mark.parametrize('spin',[0.,.3,1.2,np.pi])
def test_sky_basis_rotation_and_los_reversal(spin):
    a=project_triaxial(.8,.5,[1,2,3])
    b=project_triaxial(.8,.5,[1,2,3],spin)
    c=project_triaxial(.8,.5,[-1,-2,-3])
    assert_allclose([a.q_perp,a.b_los],[b.q_perp,b.b_los],rtol=8e-15)
    assert_allclose([a.q_perp,a.b_los],[c.q_perp,c.b_los],rtol=8e-15)
    xy=np.array([.2,-.4])
    # Same physical sky position, represented in two different sky bases.
    rotated_xy=b.sky_basis.T@(a.sky_basis@xy)
    assert_allclose(projected_radius(xy,a),projected_radius(rotated_xy,b),rtol=8e-15)


@pytest.mark.parametrize('row',ROWS[1:9])
def test_direct_untruncated_los_integral(row):
    out=project_triaxial(row['p'],row['s'],row['los'])
    # Work in rs=1, rho_s=1. Integrate the actual 3D ellipsoidal density,
    # not a completed-square or preprojected expression.
    for xy in (np.array([.2,-.35]),np.array([1.3,.7])):
        base=out.sky_basis@xy
        def density(t):
            point=base+t*out.los
            radius=np.sqrt(point@out.shape_matrix@point)
            return 1.0/(radius*(1.0+radius)**2)
        numerical=quad(density,-np.inf,np.inf,epsabs=2e-13,epsrel=2e-13,limit=300)[0]
        expected=2*out.b_los*float(nfw_f(projected_radius(xy,out)))
        assert_allclose(numerical,expected,rtol=1e-12,atol=0)


def test_circular_reference_keeps_projection_scale_and_amplitude():
    out=project_triaxial(.8,.5,[1,0,0])
    rs,ks=.37,.12
    rs_perp,ks_perp=out.circular_reference_parameters(rs,ks)
    assert not np.isclose(rs_perp,rs)
    assert not np.isclose(ks_perp,ks)
    coordinates=np.array([[.2,.1],[1.3,-.3]])
    # Express arbitrary sky positions in the projected major/minor frame.
    angle=out.major_axis_angle_rad
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    major_xy=coordinates@rotation
    zeta=area_preserving_radius(major_xy,out.q_perp)
    assert_allclose(projected_radius(coordinates,out),np.sqrt(out.b_los)*zeta,rtol=6e-15)
    assert_allclose(2*ks_perp*nfw_f(zeta/rs_perp),2*out.b_los*ks*nfw_f(projected_radius(coordinates,out)/rs),rtol=3e-14)
    # q=1 changes the elliptical radius only; keep the projected rs and ks.
    radius=area_preserving_radius(major_xy,1.)
    circular=2*ks_perp*nfw_f(radius/rs_perp)
    assert_allclose(circular,out.b_los*2*ks*nfw_f(np.sqrt(out.b_los)*radius/rs),rtol=3e-14)


@pytest.mark.parametrize('p,s',[(1.1,.5),(.5,.6),(.5,0),(.5,-.1),(np.nan,.5),(.8,np.inf)])
def test_invalid_shapes(p,s):
    with pytest.raises(ValueError): project_triaxial(p,s,[1,0,0])
    with pytest.raises(ValueError): project_triaxial_batch(p,s,[[1,0,0]])


@pytest.mark.parametrize('los',[[0,0,0],[1,2],[np.nan,0,1],[1,np.inf,0]])
def test_invalid_los(los):
    with pytest.raises(ValueError): project_triaxial(.8,.5,los)


def test_other_invalid_inputs():
    with pytest.raises(ValueError): intrinsic_shape_matrix([.8,.9],[.4,.5])
    with pytest.raises(ValueError): sky_basis_from_los([1,0,0],np.nan)
    with pytest.raises(ValueError): project_triaxial_batch([.8,.9,.9],[.5,.5,.5],[[1,0,0],[0,1,0]])
    with pytest.raises(ValueError): project_triaxial_batch(.8,.5,[])
    with pytest.raises(ValueError): area_preserving_radius([1,2],0)
    with pytest.raises(ValueError): projected_radius([1,2,3],project_triaxial(1,1,[1,0,0]))
