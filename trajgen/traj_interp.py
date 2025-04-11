import numpy as np
from typing import Literal
from scipy.spatial.transform import Rotation, Slerp
import torch
import trajgen.transforms as ttf


def traj_interp(
    traj: np.ndarray,  # (T, wxyzxyz), T is the number of waypoints
    new_start: np.ndarray,  # (wxyzxyz)
    new_end: np.ndarray = None,  # (wxyzxyz)
    mode: Literal['xyz', 'xy'] = 'xy',
    proportion: float = 1.0,
    ) -> np.ndarray:  # (T, wxyzxyz)
    """
    Interpolate a trajectory from the given start to end waypoints.
    traj: (T, wxyzxyz), T is the number of waypoints, describes how object moves in the world frame in the demonstration 
    new_start: (wxyzxyz), the start waypoint of the new trajectory
    new_end: (wxyzxyz), the end waypoint of the new trajectory
    interp_mode: interpolation mode for orientation ('linear' or 'slerp')
    proportion: the proportion of the trajectory (leading edge) that is interpolated. Default is the whole trajectory.
    """
    assert proportion > 0.0, "proportion must be greater than 0.0"
    assert proportion <= 1.0, "proportion must be less than or equal to 1.0"
    if proportion < 1.0:
        assert new_end is None, "cannot provide a new_end if leading edge proportion to interpolate is less than 1.0"
    
    T = traj.shape[0]
    
    propT = int(proportion * T)
    if proportion < 1.0:
        traj_trailing = traj[-(T-propT):,:] 
    
    # Split into orientation and position
    orig_quat = traj[:propT, :4]  # (T, wxyz)
    orig_pos = traj[:propT, 4:]  # (T, xyz)
    
    # Normalize original trajectory
    orig_start_pos = orig_pos[0]
    orig_end_pos = orig_pos[-1]
    
    if new_end is None:
        new_end = traj[:propT][-1]
    
    if mode == 'xy':
        nulls, scales = find_nulls_scales(orig_start_pos[:2], orig_end_pos[:2], new_start[4:6], new_end[4:6])
    elif mode == 'xyz':
        nulls, scales = find_nulls_scales(orig_start_pos, orig_end_pos, new_start[4:], new_end[4:])

    norm_pos = orig_pos.copy()
    
    if mode == 'xy':
        norm_pos[:, :2] -= nulls[:2]
        scaled_pos = norm_pos.copy()
        scaled_pos[:, :2] *= scales[None,:2]
        new_pos = scaled_pos.copy()
        new_pos[:, :2] += nulls[:2]
    elif mode == 'xyz':
        norm_pos -= nulls
        scaled_pos = norm_pos.copy()
        scaled_pos *= scales[None,:]
        new_pos = scaled_pos.copy()
        new_pos += nulls

    # Convert quaternions to Rotation objects
    orig_rotations = Rotation.from_quat(np.roll(orig_quat, -1, axis=1))  # Convert from wxyz to xyzw

    # Create time points for interpolation
    times = np.linspace(0, 1, propT)
    
    # Create Slerp object for original trajectory
    orig_slerp = Slerp([0, 1], Rotation.from_quat([np.roll(orig_quat[0], -1), np.roll(orig_quat[-1], -1)]))
    
    # Create Slerp object for new start/end
    new_slerp = Slerp([0, 1], Rotation.from_quat([np.roll(new_start[:4], -1), np.roll(new_end[:4], -1)]))
    
    # Interpolate rotations
    orig_rots = orig_slerp(times)
    new_rots = new_slerp(times)
    
    # Calculate rotation differences and apply to trajectory
    rot_diff = (new_rots * orig_rots.inv())
    final_rots = rot_diff * orig_rotations
    
    # Convert back to quaternions (xyzw -> wxyz)
    scaled_quat = np.roll(final_rots.as_quat(), 1, axis=1)
    
    # Combine orientation and position
    new_traj = np.concatenate([scaled_quat, new_pos], axis=1)
    
    if proportion < 1.0:
        new_traj = np.concatenate([new_traj, traj_trailing], axis=0)
    
    return new_traj


def traj_interp_batch(
    traj: np.ndarray,  # (T, wxyzxyz), T is the number of waypoints
    new_starts: np.ndarray,  # (B, wxyzxyz), B is the batch size
    new_ends: np.ndarray = None,  # (B, wxyzxyz) or None
    mode: Literal['xyz', 'xy'] = 'xy',
    proportion: float = 1.0,
    ) -> np.ndarray:  # (B, T, wxyzxyz)
    """
    Batch-interpolate multiple trajectories from the given starts to ends.
    
    Args:
        traj: (T, wxyzxyz), T is the number of waypoints, describes how object moves in the world frame in the demonstration
        new_starts: (B, wxyzxyz), B is the batch size, the start waypoints of the new trajectories
        new_ends: (B, wxyzxyz) or None, the end waypoints of the new trajectories
        mode: interpolation mode for position ('xyz' or 'xy')
        proportion: the proportion of the trajectory (leading edge) that is interpolated.
    
    Returns:
        Batch of interpolated trajectories (B, T, wxyzxyz)
    """
    assert proportion > 0.0, "proportion must be greater than 0.0"
    assert proportion <= 1.0, "proportion must be less than or equal to 1.0"
    
    # Get batch size
    batch_size = new_starts.shape[0]
    
    # If proportion is less than 1.0, we can't have new ends
    if proportion < 1.0:
        assert new_ends is None, "cannot provide new_ends if leading edge proportion to interpolate is less than 1.0"
    
    # Get number of waypoints in trajectory
    T = traj.shape[0]
    propT = int(proportion * T)
    
    # Handle trailing part of trajectory if proportion < 1.0
    if proportion < 1.0:
        traj_trailing = traj[-(T-propT):,:] 
    
    # Split into orientation and position
    orig_quat = traj[:propT, :4]  # (T, wxyz)
    orig_pos = traj[:propT, 4:]  # (T, xyz)
    
    # Get original start and end positions
    orig_start_pos = orig_pos[0]
    orig_end_pos = orig_pos[-1]
    
    # If new_ends not provided, use last waypoint of original trajectory
    if new_ends is None:
        new_ends = np.tile(traj[propT-1], (batch_size, 1))
    
    # Initialize output array for batch
    new_trajectories = np.zeros((batch_size, T, 7))
    
    # Process each item in the batch
    for b in range(batch_size):
        new_start = new_starts[b]
        new_end = new_ends[b]
        
        if mode == 'xy':
            nulls, scales = find_nulls_scales(orig_start_pos[:2], orig_end_pos[:2], new_start[4:6], new_end[4:6])
        elif mode == 'xyz':
            nulls, scales = find_nulls_scales(orig_start_pos, orig_end_pos, new_start[4:], new_end[4:])

        norm_pos = orig_pos.copy()
        
        if mode == 'xy':
            # Only transform x,y coordinates
            norm_pos[:, :2] -= nulls[:2]
            scaled_pos = norm_pos.copy()
            scaled_pos[:, :2] *= scales[None,:2]
            new_pos = scaled_pos.copy()
            new_pos[:, :2] += nulls[:2]
        elif mode == 'xyz':
            # Transform all x,y,z coordinates
            norm_pos -= nulls
            scaled_pos = norm_pos.copy()
            scaled_pos *= scales[None,:]
            new_pos = scaled_pos.copy()
            new_pos += nulls

        # Convert quaternions to Rotation objects
        orig_rotations = Rotation.from_quat(np.roll(orig_quat, -1, axis=1))  # Convert from wxyz to xyzw

        # Create time points for interpolation
        times = np.linspace(0, 1, propT)
        
        # Create Slerp object for original trajectory
        orig_slerp = Slerp([0, 1], Rotation.from_quat([np.roll(orig_quat[0], -1), np.roll(orig_quat[-1], -1)]))
        
        # Create Slerp object for new start/end
        new_slerp = Slerp([0, 1], Rotation.from_quat([np.roll(new_start[:4], -1), np.roll(new_end[:4], -1)]))
        
        # Interpolate rotations
        orig_rots = orig_slerp(times)
        new_rots = new_slerp(times)
        
        # Calculate rotation differences and apply to trajectory
        rot_diff = (new_rots * orig_rots.inv())
        final_rots = rot_diff * orig_rotations
        
        # Convert back to quaternions (xyzw -> wxyz)
        scaled_quat = np.roll(final_rots.as_quat(), 1, axis=1)
        
        # Combine orientation and position
        new_traj = np.concatenate([scaled_quat, new_pos], axis=1)
        
        # Store the leading part of the trajectory
        new_trajectories[b, :propT] = new_traj
        
        # Handle trailing part if proportion < 1.0
        if proportion < 1.0:
            new_trajectories[b, propT:] = traj_trailing
    
    return new_trajectories

def find_nulls_scales(orig_start, orig_end, new_start, new_end, epsilon=1e-10):
    """
    Find the coordinate value that maps to zero and the scale factors.
    
    Parameters:
    -----------
    orig_start : numpy.ndarray
        Original start coordinate
    orig_end : numpy.ndarray
        Original end coordinate
    new_start : numpy.ndarray
        New start coordinate after transformation
    new_end : numpy.ndarray
        New end coordinate after transformation
    epsilon : float
        Threshold for detecting near-zero scale factors
    
    Returns:
    --------
    numpy.ndarray, numpy.ndarray
        - The original coordinate value that maps to zero after the transformation
          Returns None for dimensions where there's no unique null point
        - The scale factors for each dimension
    """
    assert len(orig_start) == len(orig_end) == len(new_start) == len(new_end), "Coordinates must have the same dimension"
    nulls = []
    scales = []
    
    for i in range(len(orig_start)):
        # Case 1: If end points are fixed (pure translation at the end point)
        if abs(new_end[i] - orig_end[i]) < epsilon:
            # The scale is still calculated from the overall transformation
            scale = (new_end[i] - new_start[i]) / (orig_end[i] - orig_start[i])
            offset = new_start[i] - scale * orig_start[i]
            
            # Calculate the null point
            if abs(scale) < epsilon:
                nulls.append(None)  # No unique null point for near-zero scale
            else:
                nulls.append(orig_end[i])
            
            scales.append(scale)
            
        # Case 2: If start points are fixed (pure translation at the start point)
        elif abs(new_start[i] - orig_start[i]) < epsilon:
            # The scale is still calculated from the overall transformation
            scale = (new_end[i] - new_start[i]) / (orig_end[i] - orig_start[i])
            offset = new_start[i] - scale * orig_start[i]
            
            # Calculate the null point
            if abs(scale) < epsilon:
                nulls.append(None)  # No unique null point for near-zero scale
            else:
                nulls.append(orig_start[i])
                
            scales.append(scale)
            
        # Case 3: Pure translation (no scaling)
        elif abs((new_start[i] - orig_start[i]) - (new_end[i] - orig_end[i])) < epsilon:
            nulls.append(None)  # No unique null point for pure translation
            scales.append(1.0)  # Scale factor is 1 for pure translation
            
        # Case 4: General case - both scaling and translation
        else:
            scale = (new_end[i] - new_start[i]) / (orig_end[i] - orig_start[i])
            offset = new_start[i] - scale * orig_start[i]
            
            if abs(scale) < epsilon:
                nulls.append(None)  # No unique null point for near-zero scale
            else:
                nulls.append(-offset / scale)
                
            scales.append(scale)
            
    return np.array(nulls), np.array(scales)
    
def generate_directional_starts(single_obj_traj, batch_size, magnitude=0.07, direction_weight=0.4, perp_variation=0.15): # 0.07, 0.4, 0.15
    """
    Generate varied distribution of new start positions that tend to be in the direction away from the end of the trajectory.
    
    Args:
        single_obj_traj: Tensor of shape [T, 7] containing the original trajectory (wxyz + position)
        batch_size: Number of new start positions to generate
        magnitude: Base scale factor for the displacement
        direction_weight: How strongly to weight the directional component (0-1)
                          0 = pure random, 1 = purely in the away direction
        perp_variation: Amount of variation to add in perpendicular directions
    
    Returns:
        new_starts: Tensor of shape [batch_size, 7] with new start positions
    """
    # Extract start and end positions
    start_pos = single_obj_traj[0, 4:]
    end_pos = single_obj_traj[-1, 4:]
    
    # Calculate direction vector from end to start (i.e., away from the end)
    main_direction = start_pos - end_pos
    
    # Normalize the direction vector
    direction_norm = torch.norm(main_direction)
    if direction_norm > 1e-6:  # Avoid division by zero
        main_direction = main_direction / direction_norm
    else:
        # If start and end are too close, use a random direction
        main_direction = torch.randn(3).to(main_direction.device)
        main_direction = main_direction / torch.norm(main_direction)
    
    # Create orthogonal basis vectors to add perpendicular variation
    # Find a vector that's not parallel to main_direction
    if abs(main_direction[0]) < 0.9:
        ortho_seed = torch.tensor([1.0, 0.0, 0.0]).to(main_direction.device)
    else:
        ortho_seed = torch.tensor([0.0, 1.0, 0.0]).to(main_direction.device)
    
    # Create first perpendicular direction
    perp1 = torch.cross(main_direction, ortho_seed)
    perp1 = perp1 / torch.norm(perp1)
    
    # Create second perpendicular direction
    perp2 = torch.cross(main_direction, perp1)
    perp2 = perp2 / torch.norm(perp2)
    
    # Create batch of new start positions
    new_starts = torch.tensor(single_obj_traj[0]).repeat(batch_size, 1)
    
    # For each new start, create a more varied displacement
    for i in range(batch_size):
        # Random components (with variable strength)
        main_component_scale = torch.rand(1).item() * 1.5 + 0.5  # 0.5 to 2.0
        perp1_component_scale = (torch.rand(1).item() * 2 - 1) * perp_variation  # -perp_var to +perp_var
        perp2_component_scale = (torch.rand(1).item() * 2 - 1) * perp_variation  # -perp_var to +perp_var
        
        # Random displacement vector (for additional variation)
        random_dir = torch.randn(3).to(main_direction.device)
        random_dir = random_dir / torch.norm(random_dir)
        
        # Combine directional and random components
        displacement = (direction_weight * main_direction * main_component_scale + 
                       perp1_component_scale * perp1 +
                       perp2_component_scale * perp2 +
                       (1 - direction_weight) * random_dir)
        
        # Scale the displacement
        # Use variable magnitude to create more spread
        mag_variation = magnitude * (0.5 + torch.rand(1).item() * 1.5)  # 0.5x to 2x base magnitude
        displacement = displacement * mag_variation
        
        # Apply the displacement to the position
        new_starts[i, 4:] = start_pos + displacement

    for i in range(batch_size):
        z_rot = ttf.SO3.from_z_radians(torch.randn(1) * np.pi/10) # 8
        new_starts[i, :4] = new_starts[i, :4] + z_rot.wxyz.to(new_starts.device)

    return new_starts



def xyzw_to_wxyz(quat):
    return np.roll(quat, 1)

def wxyz_to_xyzw(quat):
    return np.roll(quat, -1)

def get_intersection_plane(p: np.ndarray, q: np.ndarray, R: float, r: float):
    """
    Find the plane of intersection between two spheres using the algebraic method.
    
    Args:
        p (np.ndarray): Center of first sphere
        q (np.ndarray): Center of second sphere
        R (float): Radius of first sphere
        r (float): Radius of second sphere
        
    Returns:
        normal (np.ndarray): Normal vector to the plane (p - q)
        d (float): Distance from origin to plane
    """
    # Normal vector is direction between centers
    normal = p - q
    
    # Calculate the plane constant from the equation https://math.stackexchange.com/questions/1628682/plane-of-intersection-of-two-spheres
    p_squared = np.dot(p, p)
    q_squared = np.dot(q, q)
    d = -(q_squared + p_squared + R**2 - r**2) / (2 * np.linalg.norm(normal))
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    return normal, d

def get_circle_parameters(p: np.ndarray, q: np.ndarray, R: float, r: float):
    """
    Find the center and radius of the intersection circle between two spheres.
    
    Args:
        p (np.ndarray): Center of first sphere
        q (np.ndarray): Center of second sphere
        R (float): Radius of first sphere
        r (float): Radius of second sphere
        
    Returns:
        center (np.ndarray): Center of intersection circle
        radius (float): Radius of intersection circle
        normal (np.ndarray): Normal vector to intersection plane
    """
    # Get the intersection plane
    normal, d = get_intersection_plane(p, q, R, r)
    
    # Calculate distance between centers
    d_centers = np.linalg.norm(p - q)
    
    # Check if spheres intersect
    if d_centers > R + r:
        raise ValueError("Spheres do not intersect")
    if d_centers < abs(R - r):
        raise ValueError("One sphere is contained within the other")
    
    # Calculate the circle center
    # It lies on the line connecting the centers at a distance h from center p
    h = (R**2 - r**2 + d_centers**2) / (2 * d_centers)
    center = p - h * normal
    
    # Calculate circle radius using Pythagorean theorem
    radius = np.sqrt(R**2 - h**2)
    
    return center, radius, normal