# PiPER · B601-RS · YAM — interactive arm viewer

A local **Viser** program showing the three manufacturer robot models at the same scale. Drag each end-effector target to move that arm with inverse kinematics.

## Open

On **macOS**, install Python 3.12+ or `uv`, then double-click **run.command** in the downloaded or cloned project folder. It starts the local server and opens http://127.0.0.1:8080. First launch installs Python dependencies into a private environment in the project folder. Leave its terminal open while using the viewer; press Control-C there to stop it.

On other systems, install Python 3.12+, create a virtual environment, install `requirements.txt`, then run `python viewer.py`. Visit http://127.0.0.1:8080 in a browser. You can select another port with `--port 8081`.

## Controls

- Drag the colored arrows at a gripper to move along an axis, or drag the little squares to move within a plane.
- The joints follow the target while respecting the **limits in the URDF**.
- Enable **Control wrist orientation** for rotation rings and pose IK.
- Drag empty space to orbit, right-drag to pan, scroll to zoom.
- Expand an arm's panel for target XYZ coordinates, gripper opening, joint sliders, and reset.
- **Base spacing (m)** changes separation between neighboring arms, initially 0.3 m.
- **Fit all arms in view** reframes the camera for the current canvas and spacing.
- **Download joint poses** exports configurations and target transforms as JSON.

The colored dot marks the actual tool position. An orange line and a residual readout show how far it is from an unreachable or joint-limited target. The local numerical solver can also stop at a local solution; an unreached target does not prove the robot can never reach that point. Reset or adjust joint sliders to try another posture.

## Models and provenance

The models and meshes are included locally, so using the viewer requires no online model requests. Original source descriptions and licenses are retained alongside the normalized `viewer.urdf` files. See `models/sources.json` for pinned Git references.

| Model | Official source | Included variant |
|---|---|---|
| AgileX PiPER | https://github.com/agilexrobotics/agx_arm_urdf | Standard PiPER with stock parallel gripper |
| Seeed reBot B601-RS | https://github.com/Seeed-Projects/reBot-DevArm | RS model with gripper |
| I2RT YAM | https://github.com/i2rt-robotics/i2rt | Standard YAM v1 with linear gripper |

PiPER's one gripper Xacro include was flattened and its package mesh paths changed to local relative paths. No source joint transforms or joint limits were changed. The Seeed and YAM viewer URDFs are copies of their originals.

Tool targets use **estimated grasp centers**, not calibrated tool frames: PiPER is `gripper_base` + 138 mm along local Z; Seeed uses the supplied `gripper_end` frame; YAM is `gripper` + 100 mm along local Z. These are easy to change in `MODELS` in `viewer.py`.

This is a **kinematic visualization**. It does not model collision avoidance, payload, motor torque, speed limits, or physical dynamics, and does not connect to real robot hardware. It is useful for comparing geometry and exploring poses, rather than certifying reachability or executing a physical trajectory. Manufacturer marketing reach measurements may use different endpoints.

## Verification

Run `python viewer.py --verify`. This checks forward kinematics against yourdfpy, analytic position Jacobians against finite differences, nearby reachable targets, position-and-orientation IK, joint limits, and clearly unreachable targets. Results are saved in `verification.json`.
