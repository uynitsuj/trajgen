import viser
import tyro
from pathlib import Path
import json
import torch
import time
from nerfstudio.utils.eval_utils import eval_setup
import numpy as np
from trajgen.traj_interp import traj_interp_batch, generate_directional_starts
import trajgen.transforms as ttf


def main(
    track_dir: Path,
    dig_config_path: Path,
    batch_size: int = 10,
    magnitude: float = 0.15,
    direction_weight: float = 0.4,
    perp_variation: float = 0.2,
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
    
    # IMPORTANT: Reorder from [batch, part, xyzwxyz] to [batch, part, wxyzxyz]
    # single_obj_traj = torch.cat([part_deltas[:,1,:4], part_deltas[:,1,4:]], dim=0).reshape(-1, 7)
    single_obj_traj = part_deltas[:,1,:]
    server = viser.ViserServer()
    
    # Visualize original trajectory
    for idx, pose in enumerate(single_obj_traj):
        server.scene.add_frame(
            name=f"original_traj/deltas/delta_{idx}",
            axes_length=0.01,
            axes_radius=0.001,
            wxyz=pose[:4].cpu().detach().numpy(), # wxyz is already first
            position=pose[4:].cpu().detach().numpy(),
        )
        
    server.scene.add_spline_catmull_rom(
            f"original_traj/catmull_rom",
            positions=single_obj_traj[:,4:].cpu().detach().numpy(), # use wxyzxyz positions
            line_width=3.0,
            curve_type="chordal",
            tension=0.1,
            color=np.random.uniform(size=3),
            segments=200,
        )
    
    # Generate directional new start positions with perpendicular variation
    new_starts = generate_directional_starts(
        single_obj_traj, 
        batch_size, 
        magnitude=magnitude, 
        direction_weight=direction_weight,
        perp_variation=perp_variation
    )
    
    # Convert tensor format for traj_interp_batch
    new_start_poses = new_starts.cpu().detach().numpy()
    
    interp_traj_frame_handles = []
    interp_traj_splines = []
    
    if len(interp_traj_frame_handles) > 0:
        for handle in interp_traj_frame_handles:
            handle.remove()
        interp_traj_frame_handles = []
        
    if len(interp_traj_splines) > 0:
        for spline in interp_traj_splines:
            spline.remove()
        interp_traj_splines = []
    
    traj = single_obj_traj.cpu().detach().numpy()  # Already in wxyzxyz format
    
    start_time = time.time()
    new_trajs = traj_interp_batch(
        traj=traj,
        new_starts=new_start_poses,
        proportion=0.6 
    )
    import pdb; pdb.set_trace()
    print(f"Time to interp {batch_size} trajectories: {time.time() - start_time}")
    
    # Visualize new trajectories
    for traj_idx, new_traj in enumerate(new_trajs):
        for idx, pose in enumerate(new_traj):
            interp_traj_frame_handles.append(server.scene.add_frame(
                name=f"interp_traj{traj_idx}/deltas/delta_{idx}",
                axes_length=0.01,
                axes_radius=0.001,
                wxyz=pose[:4],  # Take wxyz from the beginning
                position=pose[4:],  # Take xyz from the end
            ))
        
        interp_traj_splines.append(server.scene.add_spline_catmull_rom(
            f"interp_traj{traj_idx}/catmull_rom",
            positions=new_traj[:, 4:],  # Use positions from wxyzxyz format
            line_width=3.0,
            curve_type="chordal",
            tension=0.1,
            color=np.random.uniform(size=3),
            segments=200,
        ))
        
    while True:
        time.sleep(0.1)
        
    
if __name__ == "__main__":
    tyro.cli(main)