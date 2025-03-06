import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def plot_pose_axes(ax, pose, scale=0.2):
    """Plot coordinate frame axes for a given pose"""
    position = pose[:3]
    orientation = pose[3:]  # wxyz quaternion
    
    # Convert quaternion to rotation matrix
    quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
    rot_matrix = Rotation.from_quat(quat).as_matrix()
    
    # Create axes vectors
    axes = scale * rot_matrix
    
    # Plot each axis with a different color
    arrows = []
    colors = ['r', 'g', 'b']  # x = red, y = green, z = blue
    for i, color in enumerate(colors):
        arrow = ax.quiver(position[0], position[1], position[2],
                         axes[0, i], axes[1, i], axes[2, i],
                         color=color, alpha=0.8)
        arrows.append(arrow)
    return arrows

def generate_complex_trajectory(T=100, center_offset=[0, 0, 0], scale=1.0):
    """Generate a complex spiral trajectory with orientation changes"""
    t = np.linspace(0, 4*np.pi, T)
    
    # Position: Spiral with varying radius
    radius = scale * (1 + 0.3 * np.cos(3*t))
    x = radius * np.cos(t) + center_offset[0]
    y = radius * np.sin(t) + center_offset[1]
    z = scale * t/4 + 0.5 * scale * np.sin(2*t) + center_offset[2]
    
    positions = np.stack([x, y, z], axis=1)
    
    # Orientations: Complex rotation sequence
    angles_x = 0.5 * np.sin(t/2)
    angles_y = 0.3 * np.cos(t/3)  
    angles_z = t/4
    
    # Convert Euler angles to quaternions
    rots = Rotation.from_euler('xyz', np.stack([angles_x, angles_y, angles_z], axis=1))
    quats = rots.as_quat()  # scipy uses xyzw format
    quats = np.roll(quats, 1, axis=1)  # Convert to wxyz format
    
    # Combine position and orientation
    trajectory = np.concatenate([positions, quats], axis=1)
    
    return trajectory

def traj_interp(traj, new_start, new_end, interp_mode='slerp'):
    """
    Interpolate a trajectory from the given start to end waypoints.
    """
    T = traj.shape[0]
    
    # Split into position and orientation
    orig_pos = traj[:, :3]  # (T, xyz)
    orig_quat = traj[:, 3:]  # (T, wxyz)
    
    # Normalize original trajectory
    orig_start_pos = orig_pos[0]
    orig_end_pos = orig_pos[-1]
    
    # Calculate position scaling and offset
    pos_scale = np.linalg.norm(new_end[:3] - new_start[:3]) / np.linalg.norm(orig_end_pos - orig_start_pos)
    
    # Normalize and scale positions
    normalized_pos = orig_pos - orig_start_pos
    scaled_pos = normalized_pos * pos_scale
    
    # Apply new start position offset
    new_pos = scaled_pos + new_start[:3]
    
    # Handle orientations
    if interp_mode == 'linear':
        # Linear interpolation for quaternions
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
        
        # Create Slerp objects
        orig_slerp = Slerp([0, 1], Rotation.from_quat([np.roll(orig_quat[0], -1), np.roll(orig_quat[-1], -1)]))
        new_slerp = Slerp([0, 1], Rotation.from_quat([np.roll(new_start[3:], -1), np.roll(new_end[3:], -1)]))
        
        # Interpolate rotations
        orig_rots = orig_slerp(times)
        new_rots = new_slerp(times)
        
        # Calculate rotation differences and apply
        rot_diff = (new_rots * orig_rots.inv())
        final_rots = rot_diff * orig_rotations
        
        # Convert back to quaternions (xyzw -> wxyz)
        scaled_quat = np.roll(final_rots.as_quat(), 1, axis=1)
    
    # Combine position and orientation
    new_traj = np.concatenate([new_pos, scaled_quat], axis=1)
    
    return new_traj

class TrajectoryVisualizer:
    def __init__(self, orig_traj, new_traj):
        self.orig_traj = orig_traj
        self.new_traj = new_traj
        
        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize visualization elements
        self.current_frame = 0
        self.setup_plot()
        
        # Add slider
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, len(orig_traj)-1, 
                           valinit=0, valstep=1)
        self.slider.on_changed(self.update_slider)
        
        # Store plot elements that need updating
        self.orig_point = None
        self.new_point = None
        self.orig_axes = []
        self.new_axes = []
        
        # Initial update
        self.update(0)
        
    def setup_plot(self):
        """Setup static plot elements"""
        # Plot trajectories
        self.ax.plot(self.orig_traj[:, 0], self.orig_traj[:, 1], self.orig_traj[:, 2], 
                    'b-', alpha=0.5, label='Original Trajectory')
        self.ax.plot(self.new_traj[:, 0], self.new_traj[:, 1], self.new_traj[:, 2], 
                    'g-', alpha=0.5, label='New Trajectory')
        
        # Plot start and end poses
        plot_pose_axes(self.ax, self.orig_traj[0], scale=0.3)
        plot_pose_axes(self.ax, self.orig_traj[-1], scale=0.3)
        plot_pose_axes(self.ax, self.new_traj[0], scale=0.3)
        plot_pose_axes(self.ax, self.new_traj[-1], scale=0.3)
        
        # Add text annotations
        self.ax.text(self.orig_traj[0, 0], self.orig_traj[0, 1], self.orig_traj[0, 2], 
                    'Original Start', color='blue')
        self.ax.text(self.orig_traj[-1, 0], self.orig_traj[-1, 1], self.orig_traj[-1, 2], 
                    'Original End', color='blue')
        self.ax.text(self.new_traj[0, 0], self.new_traj[0, 1], self.new_traj[0, 2], 
                    'New Start', color='green')
        self.ax.text(self.new_traj[-1, 0], self.new_traj[-1, 1], self.new_traj[-1, 2], 
                    'New End', color='green')
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        
        # Set consistent axis limits
        all_points = np.vstack([self.orig_traj[:, :3], self.new_traj[:, :3]])
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        margin = 0.1 * (max_vals - min_vals)
        self.ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
        self.ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
        self.ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])
    
    def update(self, frame_idx):
        """Update dynamic plot elements"""
        # Remove previous current points and axes
        if self.orig_point is not None:
            self.orig_point.remove()
        if self.new_point is not None:
            self.new_point.remove()
        for arrow in self.orig_axes:
            arrow.remove()
        for arrow in self.new_axes:
            arrow.remove()
        
        # Plot current points
        self.orig_point = self.ax.scatter(self.orig_traj[frame_idx, 0], 
                                        self.orig_traj[frame_idx, 1], 
                                        self.orig_traj[frame_idx, 2], 
                                        c='blue', marker='o', s=100)
        
        self.new_point = self.ax.scatter(self.new_traj[frame_idx, 0], 
                                       self.new_traj[frame_idx, 1], 
                                       self.new_traj[frame_idx, 2], 
                                       c='green', marker='o', s=100)
        
        # Plot current pose axes
        self.orig_axes = plot_pose_axes(self.ax, self.orig_traj[frame_idx], scale=0.2)
        self.new_axes = plot_pose_axes(self.ax, self.new_traj[frame_idx], scale=0.2)
        
        # Update title
        self.ax.set_title(f'Trajectory Visualization (Frame {frame_idx}/{len(self.orig_traj)-1})')
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def update_slider(self, val):
        """Slider callback"""
        frame_idx = int(self.slider.val)
        self.update(frame_idx)

# Main execution
if __name__ == "__main__":
    # Generate trajectories
    T = 100
    orig_traj = generate_complex_trajectory(T, center_offset=[-2, -2, 1], scale=1.0)

    # Create new start and end poses
    new_start_pos = np.array([0.0, 2.0, 0.5])
    new_start_rot = Rotation.from_euler('xyz', [np.pi/3, np.pi/4, np.pi/6]).as_quat()
    new_start = np.concatenate([new_start_pos, np.roll(new_start_rot, 1)])

    new_end_pos = np.array([3.0, 2.5, 2.0])
    new_end_rot = Rotation.from_euler('xyz', [-np.pi/4, np.pi/3, -np.pi/2]).as_quat()
    new_end = np.concatenate([new_end_pos, np.roll(new_end_rot, 1)])

    # Generate new trajectory
    new_traj = traj_interp(orig_traj, new_start, new_end, interp_mode='slerp')

    # Create visualizer and show plot
    viz = TrajectoryVisualizer(orig_traj, new_traj)
    plt.show()