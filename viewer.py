"""Local Viser arm comparison. Uses vendored, revision-pinned manufacturer URDFs."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Model:
    key: str
    title: str
    path: str
    tip_link: str
    tip_offset: tuple[float, float, float]
    home: tuple[float, ...]
    color: tuple[int, int, int]
    base: tuple[float, float, float]

# TCPs are display grasp-center estimates; the supplied URDFs don't all define a TCP.
MODELS = [
    Model('piper', 'AgileX PiPER', 'piper/urdf/viewer.urdf', 'gripper_base', (0, 0, .138),
          (0, 1.1, -1.25, 0, .25, 0), (90, 173, 255), (0, -.3, 0)),
    Model('b601_rs', 'Seeed B601-RS', 'b601_rs/urdf/viewer.urdf', 'gripper_end', (0, 0, 0),
          (0, 1.15, 1.15, 0, .2, 0), (135, 220, 115), (0, 0, 0)),
    Model('yam', 'I2RT YAM', 'yam/viewer.urdf', 'gripper', (0, 0, .10),
          (0, 1.1, 1.2, 0, .2, 0), (241, 174, 95), (0, .3, 0)),
]

def transform(xyz=None, rpy=None):
    t = np.eye(4)
    if xyz is not None: t[:3, 3] = xyz
    if rpy is not None: t[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    return t

class Kinematics:
    """Fast URDF FK + geometric position Jacobian, independent of render meshes."""
    def __init__(self, model: Model):
        self.model = model
        root = ET.parse(ROOT/'models'/model.path).getroot()
        self.joints = {}
        by_child = {}
        for j in root.findall('joint'):
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            axis_node = j.find('axis')
            axis = np.fromstring(axis_node.get('xyz', '1 0 0'), sep=' ') if axis_node is not None else np.array([1.,0,0])
            name = j.get('name')
            data = dict(name=name, kind=j.get('type'), parent=j.find('parent').get('link'),
                        child=j.find('child').get('link'), origin=transform(xyz,rpy), axis=axis)
            self.joints[name] = data
            by_child[data['child']] = data
        self.names = [f'joint{i}' for i in range(1,7)]
        self.lower = np.array([float(root.find(f"joint[@name='{n}']/limit").get('lower')) for n in self.names])
        self.upper = np.array([float(root.find(f"joint[@name='{n}']/limit").get('upper')) for n in self.names])
        chain=[]; link=model.tip_link
        while link in by_child:
            j=by_child[link]; chain.append(j); link=j['parent']
        self.chain=chain[::-1]
        self.offset=transform(model.tip_offset)
        self.index={n:i for i,n in enumerate(self.names)}

    def fk(self, q, jacobian=False):
        t=np.eye(4); origins=[]; axes=[]
        for j in self.chain:
            t=t@j['origin']
            if j['name'] in self.index:
                origins.append(t[:3,3].copy()); axes.append(t[:3,:3]@j['axis'])
                movement=np.eye(4)
                movement[:3,:3]=Rotation.from_rotvec(j['axis']*q[self.index[j['name']]]).as_matrix()
                t=t@movement
        t=t@self.offset
        if not jacobian: return t
        jp=np.column_stack([np.cross(a,t[:3,3]-o) for a,o in zip(axes,origins)])
        return t,jp

    def solve(self, q, target, rotation=None, iterations=35):
        seed=np.clip(q,self.lower+1e-8,self.upper-1e-8)
        def residual(x):
            pose=self.fk(x)
            errors=[pose[:3,3]-target]
            if rotation is not None:
                errors.append(.16*Rotation.from_matrix(rotation@pose[:3,:3].T).as_rotvec())
            errors.append(.0003*(x-seed))
            return np.concatenate(errors)
        def jac(x):
            _,jp=self.fk(x,True)
            return np.vstack((jp,.0003*np.eye(6)))
        result=least_squares(residual,seed,bounds=(self.lower,self.upper),
            jac=jac if rotation is None else '3-point',max_nfev=iterations,
            ftol=1e-6,xtol=1e-6,gtol=1e-7)
        return result.x

def to_wxyz(matrix):
    q=Rotation.from_matrix(matrix).as_quat()
    return np.array([q[3],q[0],q[1],q[2]])

def from_wxyz(q):
    return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()

class ArmView:
    def __init__(self,server,model):
        import viser
        from viser.extras import ViserUrdf
        import yourdfpy
        self.server=server; self.model=model; self.kin=Kinematics(model)
        self.q=np.array(model.home,dtype=float);self.lock=threading.RLock()
        self.request_version=0;self.solved_version=0;self.suppress=False
        self.grip=.55
        urdf_path = ROOT/'models'/model.path
        self.urdf=yourdfpy.URDF.load(urdf_path,load_meshes=True,
            build_scene_graph=True,load_collision_meshes=False,
            filename_handler=lambda fname: str(urdf_path.parent/fname))
        if self.urdf.scene is None or len(self.urdf.scene.geometry) < 6:
            raise RuntimeError(f'Missing visual meshes for {model.title}')
        path=f'/{model.key}'
        self.base=np.array(model.base)
        self.root=server.scene.add_frame(path,position=model.base,show_axes=False)
        self.visual=ViserUrdf(server,self.urdf,root_node_name=path+'/robot')
        pose=self.kin.fk(self.q)
        self.target_position=pose[:3,3].copy();self.target_rotation=pose[:3,:3].copy()
        self.control=server.scene.add_transform_controls(path+'/target',position=pose[:3,3],
            wxyz=to_wxyz(pose[:3,:3]),scale=.16,disable_rotations=True,depth_test=False)
        self.actual=server.scene.add_icosphere(path+'/actual',radius=.011,color=model.color,position=pose[:3,3])
        self.error_line=server.scene.add_line_segments(path+'/error',points=np.array([[pose[:3,3],pose[:3,3]]]),
            colors=(255,170,60),thickness=2,thickness_units='screen')
        server.scene.add_label(path+'/name',model.title,position=(.05,0,-.04),anchor='top-center',font_screen_scale=1.2)
        self.frame=server.scene.add_frame(path+'/tcp',position=pose[:3,3],wxyz=to_wxyz(pose[:3,:3]),
            axes_length=.07,axes_radius=.002,visible=False)
        with server.gui.add_folder(model.title, expand_by_default=model.key=='piper'):
            self.status=server.gui.add_markdown('Ready')
            self.xyz=server.gui.add_vector3('Target XYZ (m)',initial_value=self.target_position,step=.01,
                hint='Relative to this arm’s base. Drag the colored arrows or enter coordinates.')
            self.orientation=server.gui.add_checkbox('Control wrist orientation',False,
                hint='Enable rotation rings and solve for both position and orientation.')
            self.gripper=server.gui.add_slider('Gripper opening',min=0.,max=1.,step=.01,initial_value=self.grip)
            self.reset=server.gui.add_button('Reset '+model.title)
            with server.gui.add_folder('Joint angles (degrees)',expand_by_default=False):
                self.sliders=[]
                for i in range(6):
                    s=server.gui.add_slider(f'Joint {i+1}',min=float(np.rad2deg(self.kin.lower[i])),
                        max=float(np.rad2deg(self.kin.upper[i])),step=.1,initial_value=float(np.rad2deg(self.q[i])))
                    self.sliders.append(s)
                    @s.on_update
                    def joint_update(event,i=i):
                        if self.suppress:return
                        with self.lock:
                            self.q[i]=np.deg2rad(event.target.value)
                            self.sync_to_actual();self.render()
        @self.control.on_update
        def drag(event):
            with self.lock:
                self.target_position=np.asarray(self.control.position).copy()
                self.target_rotation=from_wxyz(self.control.wxyz)
                self.request_version+=1
                self.suppress=True
                try:self.xyz.value=tuple(self.target_position)
                finally:self.suppress=False
        @self.xyz.on_update
        def numeric(event):
            if self.suppress:return
            with self.lock:
                self.target_position=np.array(self.xyz.value)
                self.control.position=self.target_position
                self.request_version+=1
        @self.orientation.on_update
        def change_mode(event):
            with self.lock:
                pose=self.kin.fk(self.q)
                self.target_rotation=pose[:3,:3]
                self.control.wxyz=to_wxyz(self.target_rotation)
                self.control.disable_rotations=not self.orientation.value
                self.request_version+=1
        @self.gripper.on_update
        def grip_update(event):
            with self.lock:self.grip=self.gripper.value;self.render()
        @self.reset.on_click
        def reset(event): self.reset_home()
        self.render()

    def config(self):
        cfg=dict(zip(self.kin.names,self.q))
        if self.model.key=='piper':cfg['gripper']=self.grip*.07
        elif self.model.key=='b601_rs':
            cfg['gripper_joint1']=self.grip*.05;cfg['gripper_joint2']=self.grip*.05
        else:cfg['joint7']=-self.grip*.04695;cfg['joint8']=-self.grip*.04695
        return cfg

    def sync_to_actual(self):
        pose=self.kin.fk(self.q)
        self.target_position=pose[:3,3].copy();self.target_rotation=pose[:3,:3].copy()
        self.control.position=self.target_position;self.control.wxyz=to_wxyz(self.target_rotation)
        self.suppress=True
        try:self.xyz.value=tuple(self.target_position)
        finally:self.suppress=False
        self.request_version+=1;self.solved_version=self.request_version

    def reset_home(self):
        with self.lock:
            self.q=np.array(self.model.home,dtype=float);self.sync_to_actual();self.render()

    def render(self):
        pose=self.kin.fk(self.q)
        error=float(np.linalg.norm(pose[:3,3]-self.target_position))
        angle=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(self.target_rotation@pose[:3,:3].T).as_rotvec())))
        self.visual.update_cfg(self.config())
        self.actual.position=pose[:3,3];self.frame.position=pose[:3,3];self.frame.wxyz=to_wxyz(pose[:3,:3])
        self.error_line.points=np.array([[pose[:3,3],self.target_position]])
        self.error_line.visible=error>.003
        word='✓ Reached' if error<.005 and (not self.orientation.value or angle<3) else '⚠ Limited / unreachable'
        extra=f' · orientation error **{angle:.1f}°**' if self.orientation.value else ''
        self.status.content=f'{word} · position error **{error*1000:.1f} mm**'+extra
        self.suppress=True
        try:
            for s,v in zip(self.sliders,np.rad2deg(self.q)):s.value=float(v)
        finally:self.suppress=False

    def step(self):
        with self.lock:
            if self.request_version==self.solved_version:return
            version=self.request_version;q=self.q.copy();target=self.target_position.copy()
            rotation=self.target_rotation.copy() if self.orientation.value else None
        solution=self.kin.solve(q,target,rotation)
        with self.lock:
            if version!=self.request_version:return
            self.q=solution;self.solved_version=version;self.render()


def verify():
    """Verify FK against yourdfpy, Jacobians, nearby IK, and joint-limit handling."""
    from yourdfpy import URDF
    rng=np.random.default_rng(23)
    report=[]
    for model in MODELS:
        kin=Kinematics(model);u=URDF.load(ROOT/'models'/model.path,load_meshes=False,build_scene_graph=True)
        q=np.array(model.home);worst_fk=0.;worst_jac=0.;errors=[];pose_errors=[];times=[]
        for _ in range(12):
            sample=rng.uniform(kin.lower,kin.upper)
            u.update_cfg(dict(zip(kin.names,sample)))
            ref=u.get_transform(model.tip_link)@transform(model.tip_offset)
            worst_fk=max(worst_fk,float(np.max(np.abs(ref-kin.fk(sample)))))
            pose,j=kin.fk(sample,True)
            fd=np.column_stack([(kin.fk(sample+np.eye(6)[i]*1e-6)[:3,3]-pose[:3,3])/1e-6 for i in range(6)])
            worst_jac=max(worst_jac,float(np.max(np.abs(j-fd))))
        for _ in range(15):
            goalq=np.clip(q+rng.normal(0,.09,6),kin.lower+.001,kin.upper-.001)
            goal=kin.fk(goalq);start=time.perf_counter()
            solved=kin.solve(q,goal[:3,3],iterations=60)
            times.append((time.perf_counter()-start)*1000)
            errors.append(float(np.linalg.norm(kin.fk(solved)[:3,3]-goal[:3,3])))
            assert np.all(solved>=kin.lower) and np.all(solved<=kin.upper)
            q=solved
        goal=kin.fk(np.clip(q+np.array([.03,.03,-.03,.03,.03,.03]),kin.lower+.001,kin.upper-.001))
        solved=kin.solve(q,goal[:3,3],goal[:3,:3],iterations=70)
        pose_errors.append(float(np.linalg.norm(kin.fk(solved)[:3,3]-goal[:3,3])))
        angle=float(np.linalg.norm(Rotation.from_matrix(goal[:3,:3]@kin.fk(solved)[:3,:3].T).as_rotvec()))
        unreachable=kin.solve(q,np.array([5.,5.,5.]))
        far_error=float(np.linalg.norm(kin.fk(unreachable)[:3,3]-[5,5,5]))
        assert worst_fk<1e-8,(model.key,worst_fk)
        assert worst_jac<1e-5,(model.key,worst_jac)
        assert max(errors)<.005,(model.key,errors)
        assert max(pose_errors)<.003 and angle<.02,(model.key,pose_errors,angle)
        assert far_error>1 and np.isfinite(unreachable).all()
        row=dict(robot=model.title,fk_max_error=worst_fk,jacobian_max_error=worst_jac,
                 position_ik_max_error_mm=max(errors)*1000,pose_ik_error_mm=max(pose_errors)*1000,
                 pose_ik_rotation_error_deg=np.rad2deg(angle),median_solve_ms=float(np.median(times)),
                 unreachable_error_m=far_error)
        report.append(row);print(json.dumps(row),flush=True)
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2))


def main():
    import viser
    import trimesh
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--open-browser',action='store_true');args=parser.parse_args()
    if args.verify:verify();return
    server=viser.ViserServer(host='127.0.0.1',port=args.port,label='Arm comparison')
    server.gui.configure_theme(dark_mode=True,control_width='small',
        brand_color=(115,175,255),show_logo=False,show_share_button=False)
    server.gui.main_panel.dock_right()
    server.scene.set_up_direction('+z')
    server.scene.world_axes.visible=False
    server.scene.add_grid('/floor',width=3,height=5,cell_size=.1,section_size=.5,
        cell_color=(66,76,94),section_color=(92,111,140),plane_color=(24,29,39),plane_opacity=.7,
        position=(0,0,-.018))
    server.gui.add_markdown('# Three arms, one scale\nDrag the **colored arrows** to move a gripper. Drag the **small squares** to move within a plane.\n\nDrag empty space to orbit; right-drag to pan; scroll to zoom.')
    arms=[];pedestals=[]
    for model in MODELS:
        print('Loading '+model.title,flush=True)
        pedestal=trimesh.creation.box(extents=(.25,.25,.014))
        pedestals.append(server.scene.add_mesh_simple('/pedestal_'+model.key,vertices=pedestal.vertices,faces=pedestal.faces,
            color=model.color,position=(model.base[0],model.base[1],-.007)))
        arms.append(ArmView(server,model))
    with server.gui.add_folder('Scene'):
        spacing=server.gui.add_slider('Base spacing (m)',min=.25,max=1.5,step=.025,initial_value=.3)
        fit=server.gui.add_button('Fit all arms in view')
        reset=server.gui.add_button('Reset all arms')
        frames=server.gui.add_checkbox('Show tool coordinate frames',False)
        export=server.gui.add_button('Download joint poses')
        server.gui.add_markdown('Models use metres and original URDF joint limits. Targets are grasp-center estimates. This is a kinematic viewer; collisions and loads are not simulated.\n\nSources: [AgileX](https://github.com/agilexrobotics/agx_arm_urdf) · [Seeed](https://github.com/Seeed-Projects/reBot-DevArm) · [I2RT](https://github.com/i2rt-robotics/i2rt)')
    @spacing.on_update
    def change_spacing(event):
        for i,arm in enumerate(arms):
            arm.base=np.array([0.,(i-1)*spacing.value,0.])
            arm.root.position=arm.base
            pedestals[i].position=arm.base+np.array([0,0,-.007])
    @fit.on_click
    def fit_camera(event):
        camera(event.client)
    @reset.on_click
    def reset_all(event):
        for arm in arms:arm.reset_home()
    @frames.on_update
    def show_frames(event):
        for arm in arms:arm.frame.visible=frames.value
    @export.on_click
    def download(event):
        data={}
        for arm in arms:
            with arm.lock:
                data[arm.model.key]=dict(joint_radians=arm.config(),target_xyz_m=arm.target_position.tolist(),
                    target_wxyz=to_wxyz(arm.target_rotation).tolist(),base_xyz_m=arm.base.tolist())
        event.client.send_file_download('arm-poses.json',json.dumps(data,indent=2).encode())
    @server.on_client_connect
    def camera(client):
        aspect = client.camera.aspect
        fov = np.deg2rad(48)
        distance = max(2.5, (spacing.value+.4) / (np.tan(fov/2)*aspect))
        center = np.array([.15,0,.32])
        direction = np.array([1.,0.,.75]); direction /= np.linalg.norm(direction)
        client.camera.position=center+distance*direction
        client.camera.look_at=center
        client.camera.up_direction=(0,0,1)
        client.camera.fov=fov
    for client in server.get_clients().values():camera(client)
    url=f'http://127.0.0.1:{server.get_port()}'
    print(f'READY {url}',flush=True)
    if args.open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        while True:
            start=time.monotonic()
            for arm in arms:arm.step()
            time.sleep(max(.002,1/30-(time.monotonic()-start)))
    except KeyboardInterrupt:server.stop()

if __name__=='__main__':main()
