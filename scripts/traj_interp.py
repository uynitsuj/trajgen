import numpy as np
from scipy.spatial.transform import Rotation
from autolab_core import RigidTransform
from typing import Literal

# def traj_interp(
#     traj : np.ndarray, # (T, xyzwxyz), T is the number of waypoints
#     new_start : np.ndarray, # (xyzwxyz)
#     new_end : np.ndarray, # (xyzwxyz)
#     interp_mode : str = 'linear', # 'linear' or 'slerp'
#     interp_dof_masks : np.ndarray = np.array([1, 1, 1, 1, 1, 1]), # (xyz euler angles)
# ) -> np.ndarray: # (T, xyzwxyz)
#     """
#     Interpolate a trajectory from the given start to end waypoints.
#     traj: (T, xyzwxyz), T is the number of waypoints, describes how object moves in the world frame in the demonstration 
#     new_start: (xyzwxyz), the start waypoint of the new trajectory
#     new_end: (xyzwxyz), the end waypoint of the new trajectory
#     """
#     T = len(traj)
#     new_traj = np.zeros_like(traj)
    
#     # Normalize time steps
#     t = np.linspace(0, 1, T)
    
#     # Split into position and rotation components
#     pos_orig = traj[:, :3]
#     quat_orig = traj[:, 3:7]
    
#     pos_start = new_start[:3]
#     pos_end = new_end[:3]
#     quat_start = new_start[3:7]
#     quat_end = new_end[3:7]
    
#     # Apply masks
#     pos_mask = interp_dof_masks[:3]
#     rot_mask = interp_dof_masks[3:6]
    
#     # Interpolate positions
#     for i, mask in enumerate(pos_mask):
#         if mask:
#             new_traj[:, i] = pos_start[i] + (pos_end[i] - pos_start[i]) * (traj[:, i] - traj[0, i]) / (traj[-1, i] - traj[0, i] + 1e-10)
#         else:
#             new_traj[:, i] = traj[:, i]
    
#     # Interpolate rotations
#     if interp_mode == 'linear':
#         for i in range(T):
#             if any(rot_mask):
#                 new_traj[i, 3:7] = (1 - t[i]) * quat_start + t[i] * quat_end
#                 new_traj[i, 3:7] /= np.linalg.norm(new_traj[i, 3:7])
#             else:
#                 new_traj[i, 3:7] = quat_orig[i]
#     else:  # slerp
#         for i in range(T):
#             if any(rot_mask):
#                 r_start = Rotation.from_quat(quat_start)
#                 r_end = Rotation.from_quat(quat_end)
#                 r_interp = r_start.slerp(r_end, t[i])
#                 new_traj[i, 3:7] = r_interp.as_quat()
#             else:
#                 new_traj[i, 3:7] = quat_orig[i]
    
#     return new_traj

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def traj_interp(
    traj: np.ndarray,  # (T, xyzwxyz), T is the number of waypoints
    new_start: np.ndarray,  # (xyzwxyz)
    new_end: np.ndarray = None,  # (xyzwxyz)
    interp_mode: str = 'slerp',  # 'linear' or 'slerp'
    mode: Literal['xyz', 'xy'] = 'xy',
) -> np.ndarray:  # (T, xyzwxyz)
    """
    Interpolate a trajectory from the given start to end waypoints.
    traj: (T, xyzwxyz), T is the number of waypoints, describes how object moves in the world frame in the demonstration 
    new_start: (xyzwxyz), the start waypoint of the new trajectory
    new_end: (xyzwxyz), the end waypoint of the new trajectory
    interp_mode: interpolation mode for orientation ('linear' or 'slerp')
    """
    T = traj.shape[0]
    
    # Split into position and orientation
    orig_pos = traj[:, :3]  # (T, xyz)
    orig_quat = traj[:, 3:]  # (T, wxyz)
    
    # Normalize original trajectory
    orig_start_pos = orig_pos[0]
    orig_end_pos = orig_pos[-1]
    
    if new_end is None:
        new_end = traj[-1]
    
    if mode == 'xy':
        nulls, scales = find_nulls_scales(orig_start_pos[:2], orig_end_pos[:2], new_start[:2], new_end[:2])
    elif mode == 'xyz':
        nulls, scales = find_nulls_scales(orig_start_pos, orig_end_pos, new_start[:3], new_end[:3])

    # import pdb; pdb.set_trace()
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
    
    
    # Calculate position scaling and offset
    # pos_scale = np.linalg.norm(new_end[:3] - new_start[:3]) / np.linalg.norm(orig_end_pos - orig_start_pos)
    
    # # Normalize and scale positions
    # normalized_pos = orig_pos - orig_start_pos
    # scaled_pos = normalized_pos * pos_scale
    
    # # Apply new start position offset
    # new_pos = scaled_pos + new_start[:3]
    
    # Handle orientations
    if interp_mode == 'linear':
        # Linear interpolation for quaternions (not recommended but provided as option)
        orig_start_quat = orig_quat[0]
        orig_end_quat = orig_quat[-1]
        
        # Calculate quaternion difference
        quat_diff = orig_quat - orig_start_quat
        new_quat_diff = new_end[3:] - new_start[3:]
        
        # Scale quaternion differences
        scaled_quat = orig_start_quat + (quat_diff / np.linalg.norm(quat_diff)) * np.linalg.norm(new_quat_diff)
        
    else:  # 'slerp' mode
        # Convert quaternions to Rotation objects
        orig_rotations = Rotation.from_quat(np.roll(orig_quat, -1, axis=1))  # Convert from wxyz to xyzw
        new_start_rot = Rotation.from_quat(np.roll(new_start[3:], -1))
        new_end_rot = Rotation.from_quat(np.roll(new_end[3:], -1))
        
        # Create time points for interpolation
        times = np.linspace(0, 1, T)
        
        # Create Slerp object for original trajectory
        orig_slerp = Slerp([0, 1], Rotation.from_quat([np.roll(orig_quat[0], -1), np.roll(orig_quat[-1], -1)]))
        
        # Create Slerp object for new start/end
        new_slerp = Slerp([0, 1], Rotation.from_quat([np.roll(new_start[3:], -1), np.roll(new_end[3:], -1)]))
        
        # Interpolate rotations
        orig_rots = orig_slerp(times)
        new_rots = new_slerp(times)
        
        # Calculate rotation differences and apply to trajectory
        rot_diff = (new_rots * orig_rots.inv())
        final_rots = rot_diff * orig_rotations
        
        # Convert back to quaternions (xyzw -> wxyz)
        scaled_quat = np.roll(final_rots.as_quat(), 1, axis=1)
    
    # Combine position and orientation
    new_traj = np.concatenate([new_pos, scaled_quat], axis=1)
    
    return new_traj

######## own 

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

# def find_nulls_scales(orig_start, orig_end, new_start, new_end, epsilon=1e-10):
#     """
#     Find the coordinate value that maps to zero.
    
#     Parameters:
#     -----------
#     orig_start : numpy.ndarray
#         Original start coordinate
#     orig_end : numpy.ndarray
#         Original end coordinate
#     new_start : numpy.ndarray
#         New start coordinate after transformation
#     new_end : numpy.ndarray
#         New end coordinate after transformation
#     epsilon : float
#         Threshold for detecting near-zero scale factors
    
#     Returns:
#     --------
#     numpy.ndarray
#         The original coordinate value that maps to zero after the transformation
#         Returns None for dimensions where the scale factor is near zero (also true for pure translation cases)
#     """
#     assert len(orig_start) == len(orig_end) == len(new_start) == len(new_end), "Coordinates must have the same dimension"
#     nulls = []
#     scales = []
#     for i in range(len(orig_start)):
#         if abs(new_end[i] - orig_end[i]) < epsilon:
#             nulls.append(orig_end[i])
#             scales.append(new_start[i]/orig_start[i])
#         elif new_start[i] - orig_start[i] < epsilon:
#             nulls.append(orig_start[i])
#             scales.append(new_end[i]/orig_end[i])
#         elif ((new_start[i] - orig_start[i]) - (new_end[i] - orig_end[i])) < epsilon:
#             nulls.append(None)
#             scales.append(1)
#         else:
#             scale = (new_end[i] - new_start[i]) / (orig_end[i] - orig_start[i])
#             offset = new_start[i] - scale * orig_start[i]
            
#             nulls.append(-offset / scale)
#             scales.append(scale)
#     return np.array(nulls), np.array(scales)
        
    
    
"""
1. find the mid point of the two references (A, B), call it C. From C to the start S forms a vector, which can define a plane normal that contains line AB. Find out the sign of dot(CE, CS).
2. We use similar triangle to determine the possible circle / point / no solution that the point is on given the new configuration (A', B', S')
3. ABS three points forms a plane, which we can calculate signed distance to E, scale that distance according to the distance between AB and A'B'. this should yield two point of intersections on the circle calculated in 2
4. We repeat 1) with A', B' to find a new plane and C'. Evaluate the two candidates vectors (C'E_1', C'E_2') and see which sign align with the results in 1)
"""

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

def get_new_end(
    traj: np.ndarray,  # (T, xyzwxyz), T is the number of waypoints (trajectory in old configuration)
    demo_objects_end_poses: np.ndarray,  # (N, xyzwxyz), N is the number of objects (old configuration)
    # TODO rename start_objects_poses so that it is clear that reference objects are in their end configurations
    start_objects_poses: np.ndarray,  # (N, xyzwxyz), N is the number of objects (new configuration) 
    ignore_rotation : bool = False,
) -> np.ndarray:  # (xyzwxyz)
    end_pos = traj[-1, :3]
    distances = np.linalg.norm(demo_objects_end_poses[:, :3] - end_pos, axis=1)
    object_idx = np.argmin(distances)
    ref_idx = [i for i in range(len(demo_objects_end_poses)) if i != object_idx]

    if len(ref_idx) == 1:
        # calculate the relative transform of the target object with respect to reference object in demo_objects_end_poses, apply to the reference object in start_objects_poses
        ref_idx = ref_idx[0]
        if ignore_rotation: 
            # calculate the delta in position only
            delta = demo_objects_end_poses[object_idx, :3] - demo_objects_end_poses[ref_idx, :3]
            new_target_pos = start_objects_poses[ref_idx, :3] + delta
            return np.concatenate([new_target_pos, start_objects_poses[object_idx, 3:]], axis=0)

        end_pose = RigidTransform(
            rotation=Rotation.from_quat(wxyz_to_xyzw(traj[-1, 3:])).as_matrix(),
            translation=traj[-1, :3], 
            from_frame='obj', to_frame='world'
        )
        ref_end_pose = RigidTransform(
            rotation=Rotation.from_quat(wxyz_to_xyzw(demo_objects_end_poses[ref_idx, 3:])).as_matrix(),
            translation=demo_objects_end_poses[ref_idx, :3],
            from_frame='obj', to_frame='world'
        )
        # calculate delta_tf in object frame
        delta_tf = ref_end_pose.inverse() * end_pose # from world to world
        # apply delta_tf to the start object
        ref_end_pose_new = start_objects_poses[ref_idx]
        ref_end_pose_new_tf = RigidTransform(
            rotation=Rotation.from_quat(wxyz_to_xyzw(ref_end_pose_new[3:])).as_matrix(),
            translation=ref_end_pose_new[:3],
            from_frame='obj', to_frame='world'
        )
        new_target_pose_tf = ref_end_pose_new_tf * delta_tf
        return np.concatenate([new_target_pose_tf.translation, new_target_pose_tf.quaternion], axis=0) # rigid transform uses scalar first

    # 0. define all variables 
    S, E = traj[0, :3], traj[-1, :3] # both are points
    A, B = demo_objects_end_poses[ref_idx[0], :3], demo_objects_end_poses[ref_idx[1], :3]
    # calculate C by similar triangle
    d_AS = np.linalg.norm(S - A)
    d_BS = np.linalg.norm(S - B)
    C = A + (d_AS / (d_AS + d_BS)) * (B - A) # TODO visualize this! 
    A_new, B_new = start_objects_poses[ref_idx[0], :3], start_objects_poses[ref_idx[1], :3]
    new_S = start_objects_poses[object_idx, :3]

    # 1. Find midpoints and evaluate direction sign
    assert len(ref_idx) == 2, "only support two reference objects, otherwise the problem may not have a consistent solution"
    CS, CE = S - C, E - C
    CE_CS = np.dot(CE, CS) # positive means same direction, negative means opposite direction

    # 2. Calculate similar triangle distances (EAB ~ E'A'B')
    d1 = np.linalg.norm(E - A)
    d2 = np.linalg.norm(E - B)
    old_AB = B - A
    new_AB = B_new - A_new
    scale = np.linalg.norm(new_AB) / np.linalg.norm(old_AB)
    d1_new, d2_new = d1 * scale, d2 * scale

    # 3. Find plane ABS, calculate signed distance to E 
    ABS_normal = np.cross(old_AB, CS)
    ABS_normal = ABS_normal / np.linalg.norm(ABS_normal)
    d = np.dot(CE, ABS_normal)
    scaled_d = d * scale

    # 4. find intersection points on the circle, which should have 2
    assert np.linalg.norm(new_AB) <= d1_new + d2_new, "no solution"
    if np.isclose(np.linalg.norm(new_AB), d1_new + d2_new):
        # one solution
        ratio_new_A, ratio_new_B = d1_new / (d1_new + d2_new), d2_new / (d1_new + d2_new)
        new_E = ratio_new_A * A_new + ratio_new_B * B_new
        # uses start object rotation as the end object rotation
        return np.concatenate([new_E, start_objects_poses[object_idx, 3:]], axis=0)

    # two solutions: find intersection between circle and plane
    center, radius, normal = get_circle_parameters(A_new, B_new, d1_new, d2_new)
    C_new = A_new + (d1_new / (d1_new + d2_new)) * (B_new - A_new)
    new_CS = new_S - C_new
    new_ABS_normal = np.cross(new_AB, new_CS)
    new_ABS_normal = new_ABS_normal / np.linalg.norm(new_ABS_normal)
    # plane is defined by new_ABS_normal * x - scaled_d = 0 
    # find intersection of this plane and circle to locate the two points

    # 1. Create a basis for the circle plane
    if abs(normal[2]) < abs(normal[0]):
        u = np.array([0, -normal[2], normal[1]])
    else:
        u = np.array([-normal[1], normal[0], 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # 2. Parametric equation of circle: center + r(u*cos(t) + v*sin(t))
    # Substituting into plane equation:
    # new_ABS_normal · (center + r(u*cos(t) + v*sin(t))) = scaled_d
    
    # 3. Solve for t:
    # a*cos(t) + b*sin(t) = c, where:
    a = radius * np.dot(new_ABS_normal, u)
    b = radius * np.dot(new_ABS_normal, v)
    c = scaled_d - np.dot(new_ABS_normal, center)
    
    # 4. Solution is: t = atan2(a*c ± b*sqrt(a²+b²-c²), b*c ∓ a*sqrt(a²+b²-c²))
    discriminant = a**2 + b**2 - c**2
    if discriminant < 0:
        raise ValueError("No intersection between plane and circle")
    
    sqrt_disc = np.sqrt(discriminant)
    t1 = np.arctan2(a*c + b*sqrt_disc, b*c - a*sqrt_disc)
    t2 = np.arctan2(a*c - b*sqrt_disc, b*c + a*sqrt_disc)
    
    # 5. Get the two intersection points
    E1 = center + radius * (u * np.cos(t1) + v * np.sin(t1))
    E2 = center + radius * (u * np.cos(t2) + v * np.sin(t2))
    
    # 6. Choose the point that maintains the same sign relationship
    CE1 = E1 - C_new
    CE2 = E2 - C_new
    
    dot1 = np.dot(CE1, new_CS)
    dot2 = np.dot(CE2, new_CS)
    
    # Select the point that maintains the same sign as CE_CS
    new_end_pos = E1 if (dot1 > 0) == (CE_CS > 0) else E2
    
    # 7. Handle rotation
    if ignore_rotation:
        new_end_quat = start_objects_poses[object_idx, 3:]
    else:
        # Calculate the rotation that preserves the relative orientation
        old_transform = RigidTransform(
            rotation=Rotation.from_quat(wxyz_to_xyzw(traj[-1, 3:])).as_matrix(),
            translation=E,
            from_frame='obj', to_frame='world'
        )
        old_ref_transform = RigidTransform(
            rotation=Rotation.from_quat(wxyz_to_xyzw(demo_objects_end_poses[ref_idx[0], 3:])).as_matrix(),
            translation=A,
            from_frame='obj', to_frame='world'
        )
        relative_transform = old_ref_transform.inverse() * old_transform
        
        new_ref_transform = RigidTransform(
            rotation=Rotation.from_quat(wxyz_to_xyzw(start_objects_poses[ref_idx[0], 3:])).as_matrix(),
            translation=A_new,
            from_frame='obj', to_frame='world'
        )
        new_transform = new_ref_transform * relative_transform
        new_end_quat = xyzw_to_wxyz(Rotation.from_matrix(new_transform.rotation).as_quat())
    
    # Return final pose
    return np.concatenate([new_end_pos, new_end_quat])

def get_key_frames(
    traj : np.ndarray, # (T, xyzwxyz), T is the number of waypoints
    vel_thresh : float = 0.9,
) -> np.ndarray:  # (K, xyzwxyz), K is the number of key frames 
    """
    A key frame is defined by a series of poses where the velocity of the object is slower than 90% of the rest of the trajectory
    """
    # Calculate velocities (using position only)
    velocities = np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1)
    
    # Calculate 90th percentile threshold
    thresh = np.percentile(velocities, vel_thresh * 100)
    
    # Find frames where velocity is below threshold
    # We need to handle the first and last frame specially since diff reduces length by 1
    slow_frames = np.where(velocities < thresh)[0]
    key_frames = np.unique(np.concatenate([[0], slow_frames, slow_frames + 1, [len(traj) - 1]]))
    
    return traj[key_frames]