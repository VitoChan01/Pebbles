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

def imgcreate1(xc,yc,zc):
    e1 = xc
    e2 = yc
    e3 = zc
    u1 = 0
    u2 = 0
    u3 = -1
    rays = scene.create_rays_pinhole(
        fov_deg=10,
        center=center,
        eye=[e1, e2, e3],
        up=[-e1, -e2, -e3],
        width_px=512,
        height_px=512,
    )
    sans = scene.cast_rays(rays)
    geom = sans["geometry_ids"].numpy()
    prim = sans["primitive_ids"].numpy()
    y, x = np.nonzero(prim != scene.INVALID_ID)
    rgb = np.ones((geom.shape[0], geom.shape[1], 3)) * 0.9
    rgb[y, x] = tri_rgb[prim[y, x]]

    #fg = pl.figure(1, (10.24, 10.24))
    #ax = fg.add_axes([0, 0, 1, 1])
    #ax.axis("off")
    #im = ax.imshow(rgb)
    #pl.savefig(f"outimg/rgb/07_rgb_{int(xc*100)}_{int(yc*100)}_{int(zc*100)}.png")#_{int(i*10)}_{int(j*10)}_{int(k*10)}.png")
    #pl.close("all")

    fg = pl.figure(1, (10.24, 10.24))
    ax = fg.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(sans['t_hit'].numpy())
    pl.savefig(f"outimg/distance/06_rgb_{int(xc*100)}_{int(yc*100)}_{int(zc*100)}.png")#_{int(i*10)}_{int(j*10)}_{int(k*10)}.png")
    pl.close("all")

    #dis = sans['t_hit'].numpy()
    #label = (~np.isinf(dis)).astype(int)
    #dis[np.isinf(dis)] = 0
    #dis[np.isnan(dis)] = 0
    #np.save(f'outimg/distance/07_depth_{int(xc*100)}_{int(yc*100)}_{int(zc*100)}',dis)
    #np.save(f'outimg/label/07_label_{int(xc*100)}_{int(yc*100)}_{int(zc*100)}',label)


#phi=[1,3]
#eta=[1,3]
phi = np.arange(np.pi / 48, np.pi * (2+1/48), np.pi / 24)
eta = np.arange(np.pi / 48, np.pi * (2+1/48), np.pi / 24)
#ar2=list([30])

combinations = itertools.product(list(phi),list(eta))#,ar2)
eyelist=[]
for p,e in combinations:
    eyelist.append([np.sin(e) * np.cos(p)* 0.1,
                    np.sin(e) * np.sin(p)* 0.1,
                    np.cos(e) * 0.1])
                    #f])

if __name__ == '__main__':
    with Pool() as p:
        p.starmap(imgcreate1, [(arg) for arg in eyelist])