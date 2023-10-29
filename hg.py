import math
import numpy as np
import os
import open3d as o3d
import open3d.visualization as vis
from scipy.spatial.transform import Rotation as R
from astropy.time import Time
import astropy.coordinates as coord

class MirrorElement():
    # Each mirror in the system has its dimensions,
    # device height, etc. norm0 is the norm when the
    # mirror is parallel to the wall; norm is the current
    # norm. Core position is the absolute position of the
    # mirror's center.
    def __init__(self,
                 name,
                 dimensions,
                 device_height,
                 norm0,
                 position):
        self.name = name
        self.dimension_x = dimensions[0]
        self.dimension_y = dimensions[1]
        self.device_height = device_height
        self.norm0 = norm0
        self.norm = norm0
        self.core_position = position

    def get_geometry(self):
        mirror_material = vis.Material('defaultLitSSR')
        mirror_material.scalar_properties['base_roughness'] = 0.0
        mirror_material.scalar_properties['base_reflectance'] = 1.0
        mirror_material.scalar_properties['base_clearcoat'] = 0.0
        mirror_material.scalar_properties['base_metallic'] = 1.0
        mirror_material.scalar_properties['thickness'] = 4
        mirror_material.scalar_properties['transmission']= 0.0
        mirror_material.scalar_properties['absorption_distance'] = 1

        m = Plane(self.name,
                  self.core_position,
                  self.norm,
                  self.dimension_x,
                  self.dimension_y,
                  core_anchor=[0.5,0.5],
                  core_radius=0.1 * min([self.dimension_x, self.dimension_y]))
        m.set_plane_material(mirror_material)

        return m.get_geometries()

    def set_norm(self, norm):
        self.norm = norm

    def get_reflection_rays(self, source_dir, l=10):
        source_dir = source_dir / np.linalg.norm(source_dir)
        source_apparent_point = self.core_position + source_dir * l;
        norm_apparent_point = self.core_position + self.norm * l
        outn = source_dir - 2* self.norm * np.dot(self.norm, source_dir)
        out_apparent_point = self.core_position + -outn * l
        pts = np.array([self.core_position, norm_apparent_point, source_apparent_point, out_apparent_point])
        lns = np.array([[0,1], [0,2], [0,3]])
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts),
                                        lines=o3d.utility.Vector2iVector(lns))
        return [{'name': self.name+'_reflection_rays',
                'geometry': line_set}]


class HelioGraph():
    # Some notes on the units and reference system:
    # 1. X (red) is Noth, Y (green) is Up, Z (blue) is Est.
    # 2. All lengths are in meters.
    # 3. Orietation of mirror wall (mw) and Projection wall (pw)
    #    are given in degrees with respect to North, considering
    #    the 'interesting face' that is the face with the mirror
    #    and the one on which you are projecting.
    # 4. The sun position vector should be normalized, as all
    #    the rays coming from the sun are parallel
    # 5. Mirror positions are provided in mirror wall 2D internal frame:
    #    y is equal to absolute y; x is parallel to the wall in a direction
    #    such that x cross y points to the projection wall.

    def __init__(self,
                 mw_height,
                 mw_orientation,
                 pw_distance,
                 pw_orientation,
                 location,
                 mirror_positions = [],
                 device_height = 0.01, # An orientable mirror is 10 cm from the wall surface
                 time=None,
                 ):
        self.astroloc = coord.EarthLocation(lat=location[0], lon=location[1])
        self.time = time
        self.sun_position = self.get_sun_position()

        self.mw_height = mw_height
        self.mw_orientation = mw_orientation
        self.mw_norm = np.array([np.cos(mw_orientation/180.0*np.pi),
                                 0.0,
                                 np.sin(mw_orientation/180.0*np.pi)], dtype=np.float64)
        self.pw_distance = pw_distance
        self.pw_orientation = pw_orientation
        self.pw_norm = np.array([np.cos(mw_orientation/180.0*np.pi),
                                 0.0,
                                 np.sin(mw_orientation/180.0*np.pi)], dtype=np.float64)
        self.nmirror = len(mirror_positions)
        self.mirrors = []
        for i in range(self.nmirror):
            self.mirrors += [MirrorElement("mirror_{:d}".format(i),
                                           [0.1, 0.1],
                                           device_height,
                                           self.mw_norm,
                                           np.array([0,1,0]) * mirror_positions[i][1] + \
                                           np.cross(np.array([0,1,0]), self.mw_norm) * mirror_positions[i][0] + \
                                           self.mw_norm * device_height)]

    def get_sun_position(self):
        if self.time is None:
            time = Time.now()
        else:
            time = self.time

        refsys = coord.AltAz(location=self.astroloc, obstime=time)
        sunv = np.zeros(3)
        alt = coord.get_sun(time).transform_to(refsys).alt.degree
        alt *= np.pi / 180.
        az = coord.get_sun(time).transform_to(refsys).az.degree
        az *= np.pi / 180.
        sunv[0] = np.cos(alt) * np.cos(az)
        sunv[1] = np.sin(alt)
        sunv[2] = np.cos(alt) * np.sin(az)
        return sunv

    def set_projection_points(self, pp):
        # Projection points are set in the projection wall framework
        self.tmp = []
        for i in range(self.nmirror):
            cp = self.mirrors[i].core_position
            pp_abs = self.mw_norm * self.pw_distance + \
                     np.array([0,1,0]) * pp[i][1] + \
                     np.cross(np.array([0,1,0]), self.pw_norm) * pp[i][0]

            refv =  pp_abs - cp
            refv /= np.linalg.norm(refv)
            incv = self.get_sun_position()

            hp  = (incv-refv) / 2
            nv =  hp+refv
            nv /= np.linalg.norm(nv)
            self.mirrors[i].set_norm(nv)

        return



    def geom_3d_draw(self, rays_on = False):
        ground_plane_material = vis.Material("defaultLitSSR")
        ground_plane_material.scalar_properties['roughness'] = 0.15
        ground_plane_material.scalar_properties['reflectance'] = 0.72
        ground_plane_material.scalar_properties['transmission'] = 0.6
        ground_plane_material.scalar_properties['thickness'] = 0.3
        ground_plane_material.scalar_properties['absorption_distance'] = 0.1
        ground_plane_material.vector_properties['absorption_color'] = np.array([0.82, 0.98, 0.972, 1.0])

        mp = Plane('mirror_plane', [0.,0.,0.], self.mw_norm, self.mw_height, 100, core_anchor=[0,0.5])
        pp = Plane('projection_plane', self.mw_norm * self.pw_distance, self.pw_norm, 4, 100, core_anchor=[0,0.5])

        geoms = []
        geoms += pp.get_geometries()
        geoms += mp.get_geometries()
        for m in self.mirrors:
            geoms += m.get_geometry()
            geoms += m.get_reflection_rays(self.get_sun_position())

        ax = o3d.geometry.TriangleMesh.create_coordinate_frame()
        geoms += [{'name': 'axis', 'geometry': ax}]
        geoms += self.tmp

        return geoms


class Plane():
    def __init__(self, name,
                 core_c, n, h, w,
                 core_anchor = [0.5, 0.5], thickness=1e-4, core_radius=1e-1):

        self.name = name
        self.thickness = thickness
        self.height = h
        self.width = w
        self.core_radius = core_radius
        self.core_anchor = np.array(core_anchor,dtype=np.float64)
        self.core_center = np.array(core_c, dtype=np.float64)
        self.normal = np.array(n, dtype=np.float64)
        self.normal /= np.linalg.norm(self.normal)

        self.plane = o3d.geometry.TriangleMesh.create_box(width=self.width,
                                                          height=self.thickness,
                                                          depth=self.height)
        self.core = o3d.geometry.TriangleMesh.create_sphere(self.core_radius)
        self.plane.translate([-self.core_anchor[1] * self.width,
                              -self.thickness/2,
                              -self.core_anchor[0] * self.height])
        # Rot wrt y axis
        phi = -np.arccos(np.dot(self.normal,np.array([0,1,0])))
        psi = np.arccos(np.dot(self.normal,np.array([0,0,1])))
        self.plane.translate(self.core_center)
        self.core.translate(self.core_center)

        rot = R.from_euler('xy', (phi,psi))
        self.plane.rotate(rot.as_matrix(), center=self.core_center)

        self.plane = o3d.t.geometry.TriangleMesh.from_legacy(self.plane)
        self.core = o3d.t.geometry.TriangleMesh.from_legacy(self.core)


    def add_normal(self, l=20.):
        pts = []
        pts += [self.core_center]
        pts += [self.core_center + self.normal * l]
        lines = [[0,1]]
        self.normalvecg = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts),
                                               lines=o3d.utility.Vector2iVector(lines))

    def set_plane_material(self, mat):
        self.plane.material = mat

    def get_geometries(self):
        g = [{'name': self.name+'_core',
              'geometry': self.core},
             {'name': self.name+'_plane',
              'geometry': self.plane}]
        if hasattr(self, 'normalvecg'):
            g += [{'name': self.name+'nvg',
                   'geometry' : self. normalvecg}]
        return g


if __name__ == "__main__":
    HG = HelioGraph(3,
                    180, 10, 0,
                    mirror_positions=[[0, 1],[-0.2,1], [0.2,1]],
                    location=(44.411,8.9328))
    HG.set_projection_points([[1,1],[1,1],[1,1]])
    geoms = HG.geom_3d_draw()

    vis.draw(geoms,
             bg_color=(0.9, 0.9, 0.9, 1.0),
             show_ui=True,
             width=1920,
             height=1080)
