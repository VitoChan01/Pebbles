import numpy as np
import open3d as o3d
from matplotlib import pyplot as pl
from multiprocessing import Pool
import itertools


ms = o3d.io.read_triangle_mesh("06.ply")

rgb = np.asarray(ms.vertex_colors)
tri = np.asarray(ms.triangles)
tri_rgb = np.c_[
    rgb[tri, 0].mean(axis=1),
    rgb[tri, 1].mean(axis=1),
    rgb[tri, 2].mean(axis=1)
]
center = ms.get_center()

ms = o3d.t.geometry.TriangleMesh.from_legacy(ms)

scene = o3d.t.geometry.RaycastingScene()
ix = scene.add_triangles(ms)

def imgcreate1(xc,yc,zc,fov):
    e1 = xc
    e2 = yc
    e3 = zc
    u1 = 0
    u2 = 0
    u3 = -1
    rays = scene.create_rays_pinhole(
        fov_deg=fov,
        center=center,
        eye=[e1, e2, e3],
        up=[-e1, -e2, -e3],
        width_px=1024,
        height_px=1024,
    )
    sans = scene.cast_rays(rays)
    geom = sans["geometry_ids"].numpy()
    prim = sans["primitive_ids"].numpy()
    y, x = np.nonzero(prim != scene.INVALID_ID)
    rgb = np.ones((geom.shape[0], geom.shape[1], 3)) * 0.9
    rgb[y, x] = tri_rgb[prim[y, x]]

    fg = pl.figure(1, (10.24, 10.24))
    ax = fg.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(rgb)
    pl.savefig(f"outimg/rgb/06_rgb_{int(xc*100)}_{int(yc*100)}_{int(zc*100)}_{fov}.png")#_{int(i*10)}_{int(j*10)}_{int(k*10)}.png")
    pl.close("all")

    np.save(f'outimg/distance/06_depth_{int(xc*100)}_{int(yc*100)}_{int(zc*100)}_{fov}',sans['t_hit'].numpy())


phi = np.arange(np.pi / 96, np.pi * (2+1/96), np.pi / 48)
eta = np.arange(np.pi / 96, np.pi * (2+1/96), np.pi / 48)
ar2=list([20,50,110])

combinations = itertools.product(list(phi),list(eta),ar2)
eyelist=[]
for p,e,f in combinations:
    eyelist.append([np.sin(e) * np.cos(p),
                    np.sin(e) * np.sin(p),
                    np.cos(e) * 1,
                    f])

if __name__ == '__main__':
    with Pool() as p:
        p.starmap(imgcreate1, [(arg) for arg in eyelist])