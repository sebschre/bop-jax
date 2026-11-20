"""Geometry of a 1/2<111> screw dislocation in a bcc cylinder.

Shared by ``examples/screw_dislocation.py`` and ``examples/magnetoplasticity.py``:
the anisotropic (Stroh) displacement field of a straight dislocation, a
cylinder of perfect lattice around a [111] line, and the [111] atomic columns
with the triangles a core can sit in.
"""

import numpy as np


# --- anisotropic elasticity --------------------------------------------------

def stroh_displacement(c, burgers):
    """Displacement field u(x, y) of a straight dislocation along z in an
    anisotropic medium with elastic tensor c (Stroh sextic formalism):

        u = (1/pi) Im sum_a A_a ln(x + p_a y) (L_a . b),   2 A_a . L_a = 1

    with the branch cut along the negative x axis."""
    q, r, t = c[:, 0, :, 0], c[:, 0, :, 1], c[:, 1, :, 1]
    ti = np.linalg.inv(t)
    n = np.block([[-ti @ r.T, ti], [r @ ti @ r.T - q, -r @ ti]])
    p, xi = np.linalg.eig(n)
    keep = np.argsort(-p.imag)[:3]  # the three roots in the upper half plane
    p, xi = p[keep], xi[:, keep]
    if np.min(np.abs(p[:, None] - p[None, :]) + 10 * np.eye(3)) < 1e-6:
        raise ValueError("degenerate Stroh roots (isotropic medium)")
    xi = xi / np.sqrt(2 * np.sum(xi[:3] * xi[3:], axis=0))
    a_mat, l_mat = xi[:3], xi[3:]
    coef = l_mat.T @ burgers

    def u(x, y):
        z = np.asarray(x)[..., None] + p * np.asarray(y)[..., None]
        return np.imag((np.log(z) * coef) @ a_mat.T) / np.pi

    def stress_function(x, y):  # the traction across a curve is its increment
        z = np.asarray(x)[..., None] + p * np.asarray(y)[..., None]
        return np.imag((np.log(z) * coef) @ l_mat.T) / np.pi

    # the closure failure is b and the core carries no net force
    th = np.array([-np.pi + 1e-12, np.pi - 1e-12])
    assert np.allclose(np.diff(u(np.cos(th), np.sin(th)), axis=0), burgers, atol=1e-8)
    assert np.allclose(np.diff(stress_function(np.cos(th), np.sin(th)), axis=0), 0, atol=1e-8)
    return u


# --- geometry and core analysis ----------------------------------------------

# x = [11-2], y = [-110] (normal of the (-110) glide plane), z = [111]
ROT = np.array([np.array([1, 1, -2]) / np.sqrt(6), np.array([-1, 1, 0]) / np.sqrt(2),
                np.array([1, 1, 1]) / np.sqrt(3)])


def perfect_cylinder(a, radius, length):
    n = int(np.ceil((radius + length) / a)) + 2
    g = np.stack(np.meshgrid(*[np.arange(-n, n + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    pts = a * np.concatenate([g, g + 0.5]) @ ROT.T
    rho = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[(rho < radius) & (pts[:, 2] > -1e-6) & (pts[:, 2] < length - 1e-6)]
    return pts[np.lexsort((pts[:, 2], np.hypot(pts[:, 0], pts[:, 1])))]


class Columns:
    """[111] atomic columns of the perfect cylinder and their triangles."""

    def __init__(self, ref, a, b):
        self.b = b
        key = np.round(ref[:, :2], 3)
        self.xy, self.of_atom = np.unique(key, axis=0, return_inverse=True)
        self.of_atom = self.of_atom.ravel()
        self.height = np.zeros(len(self.xy))
        self.height[self.of_atom] = np.mod(ref[:, 2], b)
        d = np.linalg.norm(self.xy[:, None] - self.xy[None], axis=-1)
        near = np.abs(d - a * np.sqrt(2 / 3)) < 0.1  # first neighbours in projection
        tri = set()
        for i in range(len(self.xy)):
            nb = np.nonzero(near[i])[0]
            for j in nb:
                for k in nb:
                    if j < k and near[j, k]:
                        tri.add(tuple(sorted((i, j, k))))
        tri = np.array(sorted(tri))
        # order counter-clockwise
        e1, e2 = self.xy[tri[:, 1]] - self.xy[tri[:, 0]], self.xy[tri[:, 2]] - self.xy[tri[:, 0]]
        cw = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] < 0
        tri[cw] = tri[cw][:, [0, 2, 1]]
        self.triangles = tri
        self.centroid = self.xy[tri].mean(axis=1)

    def easy_core_site(self):
        """Centroid of the triangle next to the axis whose column heights rise
        by b/3 counter-clockwise: the elastic field b theta / 2 pi reverses the
        helicity of that triangle, which is the compact (easy) core."""
        h = self.height[self.triangles] / self.b
        rising = np.all(np.isclose(np.mod(np.roll(h, -1, axis=1) - h, 1), 1 / 3, atol=1e-3),
                        axis=1)
        dist = np.linalg.norm(self.centroid, axis=1) + 1e3 * ~rising
        return self.centroid[np.argmin(dist)]

    def burgers_content(self, displacement_z):
        """Burgers vector (in units of b) enclosed by every triangle, from the
        differential displacements of the column-averaged u_z."""
        n = np.bincount(self.of_atom)
        u = np.bincount(self.of_atom, weights=displacement_z) / n
        t = self.triangles
        dd = u[np.roll(t, -1, axis=1)] - u[t]
        dd = dd - self.b * np.round(dd / self.b)  # differential displacement in [-b/2, b/2]
        return np.sum(dd, axis=1) / self.b

    def core(self, displacement_z):
        content = self.burgers_content(displacement_z)
        inside = np.abs(content) > 0.5
        return self.centroid[inside], content[inside]
