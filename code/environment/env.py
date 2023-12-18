"""Robot environment for the task."""

import numpy as np
import matplotlib.pyplot as plt
#import mplot3d as plt3d
# maddux for robot visualization
from maddux.robots.link import Link
from maddux.robots.arm import Arm
from maddux.environment import Environment
import time
from path import Path


class Six_Axis_Robot_Arm:
    """Class simulating the robot arm."""

    def __init__(self,
                 initial_angles: (float, float, float, float, float, float) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 starting_pos: (float, float, float) = (-500, 0, 0)) -> None:
        """Initialize robot arm.

        :param initial_angles: Tuple with the initial angles of the robot joints in degrees
        :type initial_angles: (float, float, float, float, float, float)

        :param path: Coordinates of path to draw
        :type path: list of touples of ints

        :return: None
        """
        # Create a series of links (each link has one joint)
        # (theta, offset, length, twist, q_lim=None)
        L1 = Link(0, 151.85, 0, 1.570796327, link_size=5)
        L2 = Link(0, 0, -243.55, 0, link_size=4)
        L3 = Link(0, 0, -213.2, 0, link_size=4)
        L4 = Link(0, 131.05, 0, 1.570796327, link_size=3)
        L5 = Link(0, 85.35, 0, -1.570796327, link_size=2)
        L6 = Link(0, 92.1, 0, 0, link_size=0.1)
        links = np.array([L1, L2, L3, L4, L5, L6])

        # Convert degrees of initial angles to radiants
        initial_angles_rad = np.array([self.__deg_to_rad(angle) for angle in initial_angles])
        print(initial_angles_rad)

        # Initial arm angle
        q0 = np.array(initial_angles_rad)

        # Create arm
        self.rob = Arm(links, q0, '1-link')
        # Set arm to starting position
        q = self.rob.ikine(starting_pos)
        self.rob.update_angles(q)

        self.env = Environment(dimensions=[1500.0, 1500.0, 1500.0],
                               robot=self.rob)

        # Create path for the robot
        path = Path(helix_start=starting_pos, max_distance=1)
        self.voxels, self.winning_voxels = path.get_helix_voxels()
        self.voxel_size = 1
        self.path = path.get_helix_data()

    def __deg_to_rad(self, deg: float) -> float:
        """Convert degree to radiants.

        :param deg: Degrees to convert to radiants
        :type deg: float

        :return: Radiants
        :rtype: float
        """
        return deg*np.pi/180

    def __rad_to_deg(self, rad: float) -> float:
        """Convert degree to radiants.

        :param rad: Ratiants to convert to radiants
        :type rad: float

        :return: Degrees
        :rtype: float
        """
        return rad*180/np.pi

    def get_joint_angles(self) -> (float, float, float, float, float, float):
        """Return current joint angles.

        :return: Tuple with the current angles of the robot joints in degrees
        :rtype: (float, float, float, float, float, float)

        :return: None
        """
        return np.array([self.__rad_to_deg(angle) for angle in self.rob.get_current_joint_config()])

    def set_joint_angles(self, angles: (float, float, float, float, float, float)) -> None:
        """Set joint angles.

        :param angles: Tuple with the angles for the robot joints in degrees
        :type angles: (float, float, float, float, float, float)

        :return: None
        """
        # Convert degrees of angles to radiants
        angles_rad = np.array([self.__deg_to_rad(angle) for angle in angles])
        self.rob.update_angles(angles_rad, save=True)

    def __explode(self, data):
        size = np.array(data.shape)*2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e

    def show(self, draw_path=False, draw_voxels=False, zoom_path=False) -> None:
        """Open window and draw robot arm.

        :param draw_path: Coordinates of helix to draw
        :type draw_path: list of touples of ints

        :return: None
        """
        if zoom_path is False:
            ax = self.env.plot(show=False)
        else:
            ax = self.env.plot(show=False,
                               xlim=[np.min(self.path[0])-10, np.max(self.path[0])+10],
                               ylim=[np.min(self.path[1])-10, np.max(self.path[1])+10],
                               zlim=[np.min(self.path[2])-10, np.max(self.path[2])+10],
                               )

        if draw_path is True:
            ax.plot(self.path[0], self.path[1], self.path[2], color='red')

        if draw_voxels is True:
            x = []
            y = []
            z = []
            for voxel in self.voxels:
                x.append(voxel[0])
                y.append(voxel[1])
                z.append(voxel[2])
            ax.scatter(x, y, z, marker=".", s=2, color='cyan')

            x = []
            y = []
            z = []
            for voxel in self.winning_voxels:
                x.append(voxel[0])
                y.append(voxel[1])
                z.append(voxel[2])
            ax.scatter(x, y, z, marker=".", s=2.5, color='red')

        plt.show()

    def animate(self) -> None:
        """Animate robot. Needs to be called last

        :return: None
        """
        self.env.animate()


arm = Six_Axis_Robot_Arm()
#for i in range(0, 360, 5):
#    arm.set_joint_angles((0, i, 0, 0, 0, 0))
arm.show(draw_path=True, draw_voxels=True, zoom_path=True)

# Optimize space:
# 1) All voxels with every decision at every voxel
# 2) All vocels only with evry decision at the needed voxels
# 3) Only define the coordinate of the voxels needed in an array, sort that array and use binary search for every decision in that array