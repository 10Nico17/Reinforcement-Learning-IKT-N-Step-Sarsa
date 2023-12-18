"""Robot environment for the task."""

import numpy as np
import matplotlib as plt
#import mplot3d as plt3d
# maddux for robot visualization
from maddux.robots.link import Link
from maddux.robots.arm import Arm
from maddux.environment import Environment
import time


class Six_Axis_Robot_Arm:
    """Class simulating the robot arm."""

    def __init__(self, initial_angles: (float, float, float, float, float, float) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) -> None:
        """Initialize robot arm.

        :param initial_angles: Tuple with the initial angles of the robot joints in degrees
        :type initial_angles: (float, float, float, float, float, float)

        :return: None
        """
        # Create a series of links (each link has one joint)
        # (theta, offset, length, twist, q_lim=None)
        L1 = Link(0, 15.185, 0, 1.570796327)
        L2 = Link(0, 0, -24.355, 0)
        L3 = Link(0, 0, -21.32, 0)
        L4 = Link(0, 13.105, 0, 1.570796327)
        L5 = Link(0, 8.535, 0, -1.570796327)
        L6 = Link(0, 9.21, 0, 0)
        links = np.array([L1, L2, L3, L4, L5, L6])

        # Convert degrees of initial angles to radiants
        initial_angles_rad = np.array([self.__deg_to_rad(angle) for angle in initial_angles])
        print(initial_angles_rad)

        # Initial arm angle
        q0 = np.array(initial_angles_rad)

        # Create arm
        self.rob = Arm(links, q0, '1-link')

        self.env = Environment(dimensions=[150.0, 150.0, 150.0],
                               robot=self.rob)

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
        """
        pass

    def set_joint_angles(self, angles: (float, float, float, float, float, float)) -> None:
        """Set joint angles.

        :param angles: Tuple with the angles for the robot joints in degrees
        :type angles: (float, float, float, float, float, float)
        """
        # Convert degrees of angles to radiants
        angles_rad = np.array([self.__deg_to_rad(angle) for angle in angles])
        self.rob.update_angles(angles_rad, save=True)

    def show(self) -> None:
        """Open window and draw robot arm.

        :return: None
        """
        self.env.plot()

    def animate(self) -> None:
        """Open window and draw robot arm.

        :return: None
        """
        self.env.animate()


arm = Six_Axis_Robot_Arm()
for i in range(0, 360, 5):
    arm.set_joint_angles((0, i, 0, 0, 0, 0))
arm.animate()
