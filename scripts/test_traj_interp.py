import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from traj_interp import traj_interp

def plot_pose_axes(ax, pose, scale=0.2):
    """
    Plot coordinate frame axes for a given pose
    
    Args:
        ax: Matplotlib 3D axis
        pose: Array of shape (7,) containing position (xyz) and orientation (wxyz quaternion)
        scale: Length of the coordinate axes arrows
    """
    position = pose[:3]
    orientation = pose[3:]  # wxyz quaternion
    
    # Convert quaternion to rotation matrix
    # Note: scipy expects xyzw quaternion order, so we need to reorder from wxyz
    quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
    rot_matrix = Rotation.from_quat(quat).as_matrix()
    
    # Create axes vectors
    axes = scale * rot_matrix
    
    # Plot each axis with a different color
    colors = ['r', 'g', 'b']  # x = red, y = green, z = blue
    for i, color in enumerate(colors):
        ax.quiver(position[0], position[1], position[2],
                 axes[0, i], axes[1, i], axes[2, i],
                 color=color, alpha=0.8)

def visualize_trajectories(orig_traj, new_traj):
    """
    Create an interactive visualization of original and new trajectories
    
    Args:
        orig_traj: Original trajectory, shape (T, 7) containing position (xyz) and orientation (wxyz)
        new_traj: New trajectory, shape (T, 7) containing position (xyz) and orientation (wxyz)
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    def update(frame_idx):
        ax.clear()
        
        # Plot original trajectory
        ax.plot(orig_traj[:, 0], orig_traj[:, 1], orig_traj[:, 2], 
                'b-', alpha=0.5, label='Original Trajectory')
        
        # Plot new trajectory
        ax.plot(new_traj[:, 0], new_traj[:, 1], new_traj[:, 2], 
                'g-', alpha=0.5, label='New Trajectory')
        
        # Plot start and end poses for original trajectory
        plot_pose_axes(ax, orig_traj[0], scale=0.3)
        plot_pose_axes(ax, orig_traj[-1], scale=0.3)
        
        # Plot start and end poses for new trajectory
        plot_pose_axes(ax, new_traj[0], scale=0.3)
        plot_pose_axes(ax, new_traj[-1], scale=0.3)
        
        # Add markers for current frame
        if frame_idx > 0:
            # Current pose on original trajectory
            ax.scatter(orig_traj[frame_idx, 0], orig_traj[frame_idx, 1], orig_traj[frame_idx, 2], 
                      c='blue', marker='o', s=100)
            plot_pose_axes(ax, orig_traj[frame_idx], scale=0.2)
            
            # Current pose on new trajectory
            ax.scatter(new_traj[frame_idx, 0], new_traj[frame_idx, 1], new_traj[frame_idx, 2], 
                      c='green', marker='o', s=100)
            plot_pose_axes(ax, new_traj[frame_idx], scale=0.2)
        
        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Add legend
        ax.legend()
        
        # Add text annotations for start and end points
        ax.text(orig_traj[0, 0], orig_traj[0, 1], orig_traj[0, 2], 'Original Start', color='blue')
        ax.text(orig_traj[-1, 0], orig_traj[-1, 1], orig_traj[-1, 2], 'Original End', color='blue')
        ax.text(new_traj[0, 0], new_traj[0, 1], new_traj[0, 2], 'New Start', color='green')
        ax.text(new_traj[-1, 0], new_traj[-1, 1], new_traj[-1, 2], 'New End', color='green')
        
        # Set consistent axis limits
        all_points = np.vstack([orig_traj[:, :3], new_traj[:, :3]])
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        margin = 0.1 * (max_vals - min_vals)
        ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
        ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
        ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])
        
        # Set title
        ax.set_title(f'Trajectory Visualization (Frame {frame_idx}/{len(orig_traj)-1})')
    
    # Add slider for frame control
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(orig_traj)-1, 
                   valinit=0, valstep=1)
    
    # Update function for slider
    def update_slider(val):
        update(int(slider.val))
    slider.on_changed(update_slider)
    
    # Initial plot
    update(0)
    
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate example data
    T = 50  # number of waypoints
    
    # Create a simple circular trajectory for demonstration
    t = np.linspace(0, 2*np.pi, T)
    orig_pos = np.stack([np.cos(t), np.sin(t), t/3], axis=1)
    
    # Create simple orientation change (rotation around z-axis)
    orig_quat = np.array([[np.cos(angle/2), 0, 0, np.sin(angle/2)] for angle in t])
    
    # Combine position and orientation
    orig_traj = np.concatenate([orig_pos, orig_quat], axis=1)
    
    # Create new start and end points
    new_start = np.array([1.5, 1.5, 0, 1, 0, 0, 0])  # position and quaternion
    new_end = np.array([-1.5, -1.5, 2, 0.707, 0, 0, 0.707])  # position and quaternion
    
    # Generate new trajectory
    new_traj = traj_interp(orig_traj, new_start, new_end, interp_mode='slerp')
    
    # Visualize trajectories
    visualize_trajectories(orig_traj, new_traj)