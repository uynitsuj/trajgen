import viser
import tyro
from pathlib import Path
import json
import torch
import time
from nerfstudio.utils.eval_utils import eval_setup
import numpy as np
from trajgen.traj_interp import traj_interp

def main(
    track_dir: Path,
    dig_config_path: Path,
    ):
    # Load trajectory
    if (track_dir / "cache_info.json").exists():
        cache_data = json.loads((track_dir / "cache_info.json").read_text())

        if dig_config_path is None:
            dig_config_path = Path(cache_data["dig_config_path"])
    else:
        raise FileNotFoundError("No cache info found in track.")
    
    assert dig_config_path is not None, "Must provide a dig config path."
    
    train_config, pipeline, _, _ = eval_setup(dig_config_path)
    del pipeline.garfield_pipeline
    pipeline.eval()
    
    track_data_path = track_dir / "keyframes.txt"
    if not track_data_path.exists():
        raise FileNotFoundError("No keyframes found in track.")
    
    trackdata = json.loads(track_data_path.read_text())
    part_deltas = torch.nn.Parameter(torch.tensor(trackdata["part_deltas"]).cuda())
    # T_objreg_objinit = torch.tensor(trackdata["T_objreg_objinit"]).cuda()
    # T_world_objinit = torch.tensor(trackdata["T_world_objinit"]).cuda()
    
    single_obj_traj = part_deltas[:,1,:]
    server = viser.ViserServer()
    
    for idx, pose in enumerate(single_obj_traj):
        server.scene.add_frame(
            name=f"original_traj/deltas/delta_{idx}",
            axes_length=0.01,
            axes_radius=0.001,
            wxyz=pose[:4].cpu().detach().numpy(),
            position=pose[4:].cpu().detach().numpy(),
        )

        
    server.scene.add_spline_catmull_rom(
            f"original_traj/catmull_rom",
            positions=part_deltas[:,1,4:].cpu().detach().numpy(),
            line_width=3.0,
            curve_type="chordal",
            tension=0.1,
            color=np.random.uniform(size=3),
            segments=200,
        )
    
    new_start_handle = server.scene.add_transform_controls(
        name="interp_traj/new_start",
        position=part_deltas[0,1,4:].cpu().detach().numpy(),
        wxyz=part_deltas[0,1,:4].cpu().detach().numpy(),
        scale=0.05
    )

    @new_start_handle.on_update
    def _(_):
        interp_traj_frame_handles = []
        interp_traj_spline = None
        if len(interp_traj_frame_handles) > 0:
            for interp_traj_frame_handle in interp_traj_frame_handles:
                interp_traj_frame_handle.remove()
        if interp_traj_spline is not None:
            interp_traj_spline.remove()
        new_startpose = np.array([*new_start_handle.position, *new_start_handle.wxyz])
        traj = np.concatenate([single_obj_traj[:, 4:].cpu().detach().numpy(), single_obj_traj[:, :4].cpu().detach().numpy()], axis=1)
        new_traj = traj_interp(
            traj = traj,
            new_start = new_startpose,  
            proportion=0.6 
        )
        for idx, pose in enumerate(new_traj):
            interp_traj_frame_handles.append(server.scene.add_frame(
                name=f"interp_traj/deltas/delta_{idx}",
                axes_length=0.01,
                axes_radius=0.001,
                wxyz=pose[3:],
                position=pose[:3],
            )
        )
        interp_traj_spline = server.scene.add_spline_catmull_rom(
            f"interp_traj/catmull_rom",
            positions=new_traj[:, :3],
            line_width=3.0,
            curve_type="chordal",
            tension=0.1,
            color=np.random.uniform(size=3),
            segments=200,
            )
    
    while True:
        time.sleep(0.1)
        
    
if __name__ == "__main__":
    tyro.cli(main)
