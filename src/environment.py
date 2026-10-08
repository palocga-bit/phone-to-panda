{\rtf1\ansi\ansicpg1252\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 """\
Phone-to-Panda: MuJoCo environment\
\
Defines the Panda simulation task and the deterministic\
grasp/release procedure used for the final demonstration.\
"""\
\
import numpy as np\
import mujoco\
\
\
class PandaPhoneTask:\
    """MuJoCo Panda pick-and-place environment."""\
\
    def __init__(self, xml_path):\
        self.model = mujoco.MjModel.from_xml_path(xml_path)\
        self.data = mujoco.MjData(self.model)\
\
        self.hand_id = mujoco.mj_name2id(\
            self.model,\
            mujoco.mjtObj.mjOBJ_BODY,\
            "hand"\
        )\
\
        self.cube_id = mujoco.mj_name2id(\
            self.model,\
            mujoco.mjtObj.mjOBJ_BODY,\
            "cube"\
        )\
\
        self.drop_box_id = mujoco.mj_name2id(\
            self.model,\
            mujoco.mjtObj.mjOBJ_BODY,\
            "drop_box"\
        )\
\
        self.cube_joint_id = mujoco.mj_name2id(\
            self.model,\
            mujoco.mjtObj.mjOBJ_JOINT,\
            "free"\
        )\
\
        self.cube_qpos_addr = self.model.jnt_qposadr[\
            self.cube_joint_id\
        ]\
\
    def reset(self):\
        """Reset Panda and cube to the task starting state."""\
\
        self.data.qpos[:] = 0\
        self.data.qvel[:] = 0\
\
        # Panda arm starting configuration.\
        self.data.qpos[:7] = np.array(\
            [0, -0.5, 0, -2, 0, 1.5, 0.7]\
        )\
\
        # Open gripper.\
        self.data.qpos[7:9] = 0.04\
\
        # Cube starting position.\
        self.data.qpos[\
            self.cube_qpos_addr:\
            self.cube_qpos_addr + 3\
        ] = np.array([0.40, 0.00, 0.355])\
\
        mujoco.mj_forward(\
            self.model,\
            self.data\
        )\
\
    def move_hand_to(\
        self,\
        target,\
        steps=100,\
        damping=0.05,\
        step_size=0.25\
    ):\
        """\
        Move the Panda hand towards a Cartesian target\
        using damped Jacobian inverse kinematics.\
        """\
\
        target = np.asarray(\
            target,\
            dtype=float\
        )\
\
        for _ in range(steps):\
\
            mujoco.mj_forward(\
                self.model,\
                self.data\
            )\
\
            current = self.data.xpos[\
                self.hand_id\
            ].copy()\
\
            error = target - current\
\
            if np.linalg.norm(error) < 0.005:\
                break\
\
            jac_pos = np.zeros(\
                (3, self.model.nv)\
            )\
\
            jac_rot = np.zeros(\
                (3, self.model.nv)\
            )\
\
            mujoco.mj_jacBody(\
                self.model,\
                self.data,\
                jac_pos,\
                jac_rot,\
                self.hand_id\
            )\
\
            J = jac_pos[:, :7]\
\
            JJt = J @ J.T\
\
            dq = (\
                J.T @\
                np.linalg.solve(\
                    JJt +\
                    damping**2 *\
                    np.eye(3),\
                    error\
                )\
            )\
\
            dq = np.clip(\
                dq,\
                -step_size,\
                step_size\
            )\
\
            self.data.qpos[:7] += dq\
\
            for j in range(7):\
                self.data.qpos[j] = np.clip(\
                    self.data.qpos[j],\
                    self.model.jnt_range[j, 0],\
                    self.model.jnt_range[j, 1]\
                )\
\
            mujoco.mj_forward(\
                self.model,\
                self.data\
            )\
\
    def set_gripper(self, closed):\
        """Open or close the Panda gripper."""\
\
        value = 0.0 if closed else 0.04\
\
        self.data.qpos[7:9] = value\
\
        mujoco.mj_forward(\
            self.model,\
            self.data\
        )\
\
    def set_cube_position(self, position):\
        """Set cube Cartesian position."""\
\
        self.data.qpos[\
            self.cube_qpos_addr:\
            self.cube_qpos_addr + 3\
        ] = np.asarray(\
            position,\
            dtype=float\
        )\
\
        self.data.qvel[:] = 0\
\
        mujoco.mj_forward(\
            self.model,\
            self.data\
        )\
\
    def get_hand_position(self):\
        """Return current Panda hand position."""\
\
        mujoco.mj_forward(\
            self.model,\
            self.data\
        )\
\
        return self.data.xpos[\
            self.hand_id\
        ].copy()\
\
    def get_cube_position(self):\
        """Return current cube position."""\
\
        mujoco.mj_forward(\
            self.model,\
            self.data\
        )\
\
        return self.data.xpos[\
            self.cube_id\
        ].copy()}