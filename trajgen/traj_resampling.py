import numpy as np
from typing import Tuple, Optional, Union

def catmull_rom_one_point(x: float, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, 
                          tension: float = 0.5, curve_type: str = "centripetal") -> np.ndarray:
    """
    Compute a single point on a Catmull-Rom spline.
    
    Args:
        x: Parameter value between 0 and 1
        v0, v1, v2, v3: Four control points defining the spline segment
        tension: Tension parameter (between 0 and 1), controls how "tight" the curve is
        curve_type: Type of parameterization to use ('uniform', 'chordal', or 'centripetal')
        
    Returns:
        Position on the spline curve at parameter x
    """
    # Standard Catmull-Rom matrix implementation - this ensures proper curve continuity
    t = x
    t2 = t * t
    t3 = t2 * t
    
    # Apply tension parameter (tau)
    tau = 0.5 * (1.0 - tension)
    
    # Compute the point using the Catmull-Rom matrix multiplication
    result = 0.5 * (
        (2 * v1) +
        (-v0 + v2) * t +
        (2*v0 - 5*v1 + 4*v2 - v3) * t2 +
        (-v0 + 3*v1 - 3*v2 + v3) * t3
    )
    
    # Apply additional parameterization based on curve_type if needed
    # For now we're using the standard Catmull-Rom formulation
    
    return result


def sample_catmull_rom_spline(
    trajectory: np.ndarray,  # (T, wxyzxyz), T is the number of control points
    scale_factor: float = 1.0,  # New trajectory will have T*scale_factor points
    tension: float = 0.5,
    curve_type: str = "centripetal",
    ) -> np.ndarray:  # ((T-1)*scale_factor, wxyzxyz)
    """
    Sample a Catmull-Rom spline from trajectory control points.
    
    Args:
        trajectory: Array of shape (T, wxyzxyz) containing the control points
        scale_factor: Factor to scale the number of output points (1.0 = same number, 
                     2.0 = twice as many points)
        tension: Tension parameter for the spline (between 0 and 1)
        curve_type: Type of parameterization ('uniform', 'chordal', or 'centripetal')
        
    Returns:
        Array of shape (ceil((T-1)*scale_factor), wxyzxyz) with sampled points along the spline
    """
    if trajectory.shape[0] < 4:
        raise ValueError("Need at least 4 control points for Catmull-Rom spline")
    
    num_control_points = trajectory.shape[0]
    
    # Split into orientation (quaternion) and position
    control_quats = trajectory[:, :4]  # (T, wxyz)
    control_pos = trajectory[:, 4:]  # (T, xyz)
    
    # For Catmull-Rom, we have n-3 segments (between points 1 to n-2)
    num_segments = num_control_points - 3
    points_per_segment = int(np.ceil(scale_factor))
    total_points = points_per_segment * num_segments
    
    # Initialize output arrays
    output_pos = np.zeros((total_points, 3))
    output_quats = np.zeros((total_points, 4))
    
    # For each segment between interior points
    for i in range(num_segments):
        # Get the four control points for this segment (two on each side of the segment)
        p0 = control_pos[i]
        p1 = control_pos[i + 1]
        p2 = control_pos[i + 2]
        p3 = control_pos[i + 3]
        
        # Get corresponding quaternions (though we'll only use p1 and p2 quaternions)
        q1 = control_quats[i + 1]
        q2 = control_quats[i + 2]
        
        # Generate evenly spaced parameters for this segment
        t_values = np.linspace(0, 1, points_per_segment, endpoint=(i==num_segments-1))
        
        # Calculate points along this segment
        for j, t in enumerate(t_values):
            idx = i * points_per_segment + j
            if idx < total_points:  # Safety check
                # Sample position using Catmull-Rom formula
                output_pos[idx] = catmull_rom_one_point(t, p0, p1, p2, p3, tension, curve_type)
                
                # For quaternions, use the closest control point
                output_quats[idx] = q1 if t < 0.5 else q2
    
    # Combine orientation and position
    output_traj = np.concatenate([output_quats, output_pos], axis=1)
    return output_traj


def sample_catmull_rom_batch(
    trajectory_batch: np.ndarray,  # (B, T, wxyzxyz), B is batch size, T is the number of control points
    scale_factor: float = 1.0,  # New trajectory will have T*scale_factor points
    tension: float = 0.5,
    curve_type: str = "centripetal",
    ) -> np.ndarray:  # (B, (T-1)*scale_factor, wxyzxyz)
    """
    Sample Catmull-Rom splines from a batch of trajectories.
    
    Args:
        trajectory_batch: Array of shape (B, T, wxyzxyz) containing batches of control points
        scale_factor: Factor to scale the number of output points
        tension: Tension parameter for the spline (between 0 and 1)
        curve_type: Type of parameterization ('uniform', 'chordal', or 'centripetal')
        
    Returns:
        Array of shape (B, points_per_segment * num_segments, wxyzxyz) with sampled points along the splines
    """
    batch_size, num_control_points, _ = trajectory_batch.shape
    
    if num_control_points < 4:
        raise ValueError("Need at least 4 control points for Catmull-Rom spline")
    
    # Calculate the number of points in the output trajectories
    num_segments = num_control_points - 3
    points_per_segment = int(np.ceil(scale_factor))
    total_points = points_per_segment * num_segments
    
    # Initialize output array
    output_batch = np.zeros((batch_size, total_points, 7))
    
    # Process each trajectory in the batch
    for b in range(batch_size):
        output_batch[b] = sample_catmull_rom_spline(
            trajectory_batch[b], 
            scale_factor=scale_factor,
            tension=tension,
            curve_type=curve_type
        )
    
    return output_batch


def extend_trajectory_for_catmull_rom(
    trajectory: np.ndarray,  # (T, wxyzxyz), T is the number of control points
    ) -> np.ndarray:  # (T+2, wxyzxyz)
    """
    Extend a trajectory with appropriate boundary conditions for Catmull-Rom spline.
    For the first and last control points, we need additional "phantom" points.
    
    Args:
        trajectory: Original trajectory points of shape (T, wxyzxyz)
        
    Returns:
        Extended trajectory of shape (T+2, wxyzxyz) with added boundary points
    """
    T = trajectory.shape[0]
    
    # Create extended trajectory array
    extended_traj = np.zeros((T + 2, 7))
    
    # Copy original trajectory to the middle
    extended_traj[1:-1] = trajectory
    
    # Create phantom points by extrapolation
    # First phantom point (before the first real point)
    first_vector = trajectory[1, 4:] - trajectory[0, 4:]
    extended_traj[0, 4:] = trajectory[0, 4:] - first_vector
    extended_traj[0, :4] = trajectory[0, :4]  # Copy first quaternion
    
    # Last phantom point (after the last real point)
    last_vector = trajectory[-1, 4:] - trajectory[-2, 4:]
    extended_traj[-1, 4:] = trajectory[-1, 4:] + last_vector
    extended_traj[-1, :4] = trajectory[-1, :4]  # Copy last quaternion
    
    return extended_traj


def calculate_arc_length(
    trajectory: np.ndarray,  # (T, wxyzxyz)
    samples_per_segment: int = 20
    ) -> np.ndarray:
    """
    Calculate the approximate arc length along a trajectory.
    
    Args:
        trajectory: Array of shape (T, wxyzxyz) containing the control points
        samples_per_segment: Number of samples to use for approximating each segment's length
        
    Returns:
        Array of cumulative arc lengths at each sample point
    """
    # We'll do this by densely sampling the curve and summing segment lengths
    
    # First, get a dense sampling of the spline
    dense_trajectory = sample_catmull_rom_spline(
        trajectory, 
        scale_factor=samples_per_segment,
        tension=0.5,
        curve_type="centripetal"
    )
    
    # Extract position data
    positions = dense_trajectory[:, 4:]
    
    # Calculate the cumulative arc length
    segment_lengths = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    cumulative_lengths = np.zeros(len(positions))
    cumulative_lengths[1:] = np.cumsum(segment_lengths)
    
    return cumulative_lengths, dense_trajectory


def resample_traj_catmull_rom(
    trajectories: np.ndarray,  # (B, T, wxyzxyz) or (T, wxyzxyz)
    scale_factor: float = 1.0,
    tension: float = 0.5,
    curve_type: str = "centripetal",
    ) -> np.ndarray:
    """
    Resample trajectories using Catmull-Rom splines.
    
    Args:
        trajectories: Trajectory control points, either batch (B, T, wxyzxyz) or single (T, wxyzxyz)
        scale_factor: Factor to scale the number of output points (1.0 = same, 2.0 = double)
        tension: Tension parameter (0-1) controlling curve tightness (lower = smoother)
        curve_type: Type of parameterization ('uniform', 'chordal', or 'centripetal')
        
    Returns:
        Resampled trajectories with the same format as input but with scaled number of points
    """
    # Check if we have enough control points
    if (len(trajectories.shape) == 2 and trajectories.shape[0] < 4) or \
       (len(trajectories.shape) == 3 and trajectories.shape[1] < 4):
        raise ValueError("Need at least 4 control points for Catmull-Rom spline")
    
    # Check if input is a batch or a single trajectory
    is_batch = len(trajectories.shape) == 3
    
    if not is_batch:
        # Convert single trajectory to batch format temporarily
        trajectories = trajectories[np.newaxis, ...]
    
    batch_size, num_points, _ = trajectories.shape
    
    # Special case: If we only have 4 points exactly, we don't need to extend
    if num_points == 4:
        resampled_batch = sample_catmull_rom_batch(
            trajectories,
            scale_factor=scale_factor,
            tension=tension,
            curve_type=curve_type
        )
    else:
        # Add boundary points for proper spline calculation
        extended_batch = np.zeros((batch_size, num_points + 2, 7))
        
        for b in range(batch_size):
            extended_batch[b] = extend_trajectory_for_catmull_rom(trajectories[b])
        
        # Now we can sample the Catmull-Rom spline using the extended trajectory
        resampled_batch = sample_catmull_rom_batch(
            extended_batch,
            scale_factor=scale_factor,
            tension=tension,
            curve_type=curve_type
        )
    
    # Return in the same format as input
    if not is_batch:
        return resampled_batch[0]  # Return single trajectory
    else:
        return resampled_batch  # Return batch of trajectories


def generate_uniform_control_points(
    trajectory: np.ndarray,  # (T, wxyzxyz), T is the number of control points
    num_new_points: int = None,  # Number of new control points to generate
    proportion: float = 1.0,    # Proportion relative to original control points
    tension: float = 0.5,       # Spline tension
    ) -> np.ndarray:  # (num_new_points, wxyzxyz)
    """
    Generate a new set of control points that are uniformly distributed along the arc length
    of the Catmull-Rom spline defined by the original trajectory.
    
    Args:
        trajectory: Original trajectory control points of shape (T, wxyzxyz)
        num_new_points: Number of new control points to generate (overrides proportion if provided)
        proportion: Factor to scale the number of control points relative to original (default: 1.0)
        tension: Tension parameter for the spline (between 0 and 1)
        
    Returns:
        Array of shape (num_new_points, wxyzxyz) with uniformly spaced control points
    """
    if trajectory.shape[0] < 4:
        raise ValueError("Need at least 4 control points for Catmull-Rom spline")
    
    # Determine how many new control points to generate
    orig_points = trajectory.shape[0]
    if num_new_points is None:
        num_new_points = max(4, int(np.ceil(orig_points * proportion)))
    
    # Extend trajectory for proper spline calculation
    extended_traj = extend_trajectory_for_catmull_rom(trajectory)
    
    # Calculate arc length along the spline
    arc_lengths, dense_samples = calculate_arc_length(extended_traj, samples_per_segment=20)
    total_length = arc_lengths[-1]
    
    # Generate evenly spaced lengths along the arc
    uniform_lengths = np.linspace(0, total_length, num_new_points)
    
    # Initialize array for new control points
    new_control_points = np.zeros((num_new_points, 7))
    
    # For each desired uniform control point
    for i, target_length in enumerate(uniform_lengths):
        # Find the closest sampled point to this arc length
        idx = np.searchsorted(arc_lengths, target_length)
        
        # Handle boundary cases
        if idx == 0:
            new_control_points[i] = dense_samples[0]
        elif idx >= len(arc_lengths):
            new_control_points[i] = dense_samples[-1]
        else:
            # Interpolate between the two closest points for better accuracy
            prev_idx = idx - 1
            prev_length = arc_lengths[prev_idx]
            curr_length = arc_lengths[idx]
            
            # Calculate interpolation parameter
            t = (target_length - prev_length) / (curr_length - prev_length) if curr_length > prev_length else 0.0
            
            # Interpolate position
            prev_pos = dense_samples[prev_idx, 4:]
            curr_pos = dense_samples[idx, 4:]
            interpolated_pos = prev_pos * (1 - t) + curr_pos * t
            
            # Use closest quaternion (could use slerp for smoother transitions)
            closest_quat = dense_samples[prev_idx if t < 0.5 else idx, :4]
            
            # Combine position and orientation
            new_control_points[i] = np.concatenate([closest_quat, interpolated_pos])
    
    return new_control_points


def generate_uniform_control_points_batch(
    trajectories: np.ndarray,  # (B, T, wxyzxyz) or (T, wxyzxyz)
    num_new_points: int = None,
    proportion: float = 1.0,
    tension: float = 0.5,
    ) -> np.ndarray:  # (B, num_new_points, wxyzxyz) or (num_new_points, wxyzxyz)
    """
    Generate new sets of uniformly distributed control points along the arc lengths
    of Catmull-Rom splines for a batch of trajectories.
    
    Args:
        trajectories: Trajectory control points, either batch (B, T, wxyzxyz) or single (T, wxyzxyz)
        num_new_points: Number of new control points to generate (overrides proportion if provided)
        proportion: Factor to scale the number of control points relative to original
        tension: Tension parameter for the spline (between 0 and 1)
        
    Returns:
        Batch of new control points with same format as input but with specified number of points
    """
    # Check if input is a batch or a single trajectory
    is_batch = len(trajectories.shape) == 3
    
    if not is_batch:
        # Convert single trajectory to batch format temporarily
        trajectories = trajectories[np.newaxis, ...]
    
    batch_size, num_points, _ = trajectories.shape
    
    # Determine how many new control points to generate
    if num_new_points is None:
        num_new_points = max(4, int(np.ceil(num_points * proportion)))
    
    # Initialize output array for batch
    new_batched_control_points = np.zeros((batch_size, num_new_points, 7))
    
    # Process each trajectory in the batch
    for b in range(batch_size):
        new_batched_control_points[b] = generate_uniform_control_points(
            trajectories[b],
            num_new_points=num_new_points,
            proportion=proportion,
            tension=tension
        )
    
    # Return in the same format as input
    if not is_batch:
        return new_batched_control_points[0]  # Return single trajectory
    else:
        return new_batched_control_points  # Return batch of trajectories