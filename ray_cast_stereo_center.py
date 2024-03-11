import numpy as np
import open3d as o3d
from matplotlib import pyplot as pl
from tqdm import trange


ms02 = o3d.io.read_triangle_mesh("02.ply")
ms03 = o3d.io.read_triangle_mesh("03.ply")
ms04 = o3d.io.read_triangle_mesh("04.ply")
ms05 = o3d.io.read_triangle_mesh("05.ply")
ms06 = o3d.io.read_triangle_mesh("06.ply")
ms07 = o3d.io.read_triangle_mesh("07.ply")
ms08 = o3d.io.read_triangle_mesh("08.ply")

ms02 = ms02.translate([0, 0, -0.12])
ms02 = ms02.scale(2, center=ms02.get_center())
ms03 = ms03.translate([0, 0, -0.1])
ms03 = ms03.scale(1.5, center=ms02.get_center())
ms04 = ms04.translate([0, 0, -0.12])
ms04 = ms04.scale(3, center=ms04.get_center())
ms08 = ms08.scale(1.5, center=ms08.get_center())

center = []
tcolor = []
for ms in (ms02, ms03, ms04, ms05, ms06, ms07, ms08):
    rgb = np.asarray(ms.vertex_colors)
    tri = np.asarray(ms.triangles)
    rgb = np.c_[
        rgb[tri, 0].mean(axis=1),
        rgb[tri, 1].mean(axis=1),
        rgb[tri, 2].mean(axis=1)
    ]
    tcolor.append(rgb)
    center.append(ms.get_center())
center = np.array(center).mean(axis=0)

ms02 = o3d.t.geometry.TriangleMesh.from_legacy(ms02)
ms03 = o3d.t.geometry.TriangleMesh.from_legacy(ms03)
ms04 = o3d.t.geometry.TriangleMesh.from_legacy(ms04)
ms05 = o3d.t.geometry.TriangleMesh.from_legacy(ms05)
ms06 = o3d.t.geometry.TriangleMesh.from_legacy(ms06)
ms07 = o3d.t.geometry.TriangleMesh.from_legacy(ms07)
ms08 = o3d.t.geometry.TriangleMesh.from_legacy(ms08)

scene = o3d.t.geometry.RaycastingScene()
id02 = scene.add_triangles(ms02)
id03 = scene.add_triangles(ms03)
id04 = scene.add_triangles(ms04)
id05 = scene.add_triangles(ms05)
id06 = scene.add_triangles(ms06)
id07 = scene.add_triangles(ms07)
id08 = scene.add_triangles(ms08)

# eye_c are coordinates for the point between cameras
# spherical coordinate ranges
rho = 1
phi = np.arange(np.pi / 24, np.pi * (2+1/24), np.pi / 12)
eta = np.array((1/2, 1/3, 1/6)) * np.pi
one = np.ones(phi.shape)

# convert to cartesian coordinates
eye_c = np.c_[np.sin(eta[0]) * np.cos(phi),
              np.sin(eta[0]) * np.sin(phi),
              np.cos(eta[0]) * one]
for i in (1, 2):
    eye_c = np.vstack((eye_c,
                       np.c_[np.sin(eta[i]) * np.cos(phi),
                             np.sin(eta[i]) * np.sin(phi),
                             np.cos(eta[i]) * one]
                       ))
eye_c *= rho
eye_c += center

# base line distance between cameras
base = 0.05

# line equation
ratios = (center[0] - eye_c[:, 0]) / (center[1] - eye_c[:, 1])
print(center[1], eye_c[:2, 1])

x = np.sqrt(base * base / 4 / (1 + ratios * ratios))
y = -x * ratios
z = np.zeros(x.shape)
eye_a = np.c_[x, y, z] + eye_c
eye_b = np.c_[-x, -y, z] + eye_c

f = np.tan(15 * np.pi / 180) * 6000 / 2
print("focal length:", f)
for i in trange(eye_a.shape[0]):
    # camera a
    rays = scene.create_rays_pinhole(
        fov_deg=30,
        center=center,
        eye=eye_a[i],
        up=[0, 0, -1],
        width_px=6000,
        height_px=4000,
    )
    ans = scene.cast_rays(rays)
    geom = ans["geometry_ids"].numpy()
    prim = ans["primitive_ids"].numpy()
    y, x = np.nonzero(prim != scene.INVALID_ID)
    gyx = geom[y, x]
    ug = np.unique(gyx)
    rgb = np.nan * np.ones((geom.shape[0], geom.shape[1], 3))
    for g in ug:
        sl = np.where(gyx == g)[0]
        tc = tcolor[g]
        rgb[y[sl], x[sl]] = tc[prim[y[sl], x[sl]]]
    fg = pl.figure(1, (60, 40))
    ax = fg.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(rgb)
    pl.savefig("a_f%0.4f_d%0.3f_img%03d.jpg" % (f, base, i))
    pl.close("all")
    # camera b
    rays = scene.create_rays_pinhole(
        fov_deg=30,
        center=center,
        eye=eye_b[i],
        up=[0, 0, -1],
        width_px=6000,
        height_px=4000,
    )
    ans = scene.cast_rays(rays)
    geom = ans["geometry_ids"].numpy()
    prim = ans["primitive_ids"].numpy()
    y, x = np.nonzero(prim != scene.INVALID_ID)
    gyx = geom[y, x]
    ug = np.unique(gyx)
    rgb = np.nan * np.ones((geom.shape[0], geom.shape[1], 3))
    for g in ug:
        sl = np.where(gyx == g)[0]
        tc = tcolor[g]
        rgb[y[sl], x[sl]] = tc[prim[y[sl], x[sl]]]
    fg = pl.figure(1, (60, 40))
    ax = fg.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(rgb)
    pl.savefig("b_f%0.4f_d%0.3f_img%03d.jpg" % (f, base, i))
    pl.close("all")
