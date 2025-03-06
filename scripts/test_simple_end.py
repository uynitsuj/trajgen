import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from traj_interp import get_new_end

@dataclass
class RigidTransform:
    rotation: Rotation
    translation: np.ndarray
    from_frame: str = 'obj'
    to_frame: str = 'world'
    
    def inverse(self):
        inv_rotation = self.rotation.inv()
        inv_translation = -inv_rotation.apply(self.translation)
        return RigidTransform(
            rotation=inv_rotation,
            translation=inv_translation,
            from_frame=self.to_frame,
            to_frame=self.from_frame
        )
    
    def __mul__(self, other):
        """Compose two transforms: self * other"""
        rotation = self.rotation * other.rotation
        translation = self.translation + self.rotation.apply(other.translation)
        return RigidTransform(
            rotation=rotation,
            translation=translation,
            from_frame=other.from_frame,
            to_frame=self.to_frame
        )
    
    @property
    def quaternion(self):
        return self.rotation.as_quat()  # returns scalar-last format

def plot_pose_axes(ax, pose, scale=0.2):
    """Plot coordinate frame axes for a given pose"""
    position = pose[:3]
    orientation = pose[3:]  # wxyz quaternion
    
    quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
    rot_matrix = Rotation.from_quat(quat).as_matrix()
    axes = scale * rot_matrix
    
    colors = ['r', 'g', 'b']
    for i, color in enumerate(colors):
        ax.quiver(position[0], position[1], position[2],
                 axes[0, i], axes[1, i], axes[2, i],
                 color=color, alpha=0.8)

def generate_test_data(test_type="single_reference"):
    """Generate test data for different scenarios"""
    if test_type == "single_reference":
        # Test with single reference, pure translation
        demo_end = np.array([
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Target
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Reference
        ])
        
        start_poses = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Target start
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Reference
        ])
        
        # Create trajectory with multiple waypoints
        num_points = 20
        t = np.linspace(0, 1, num_points)
        x = np.zeros_like(t)
        y = t
        z = np.zeros_like(t)
        quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_points, 1))
        traj = np.column_stack((x, y, z, quat))
        
    elif test_type == "single_reference_rotation":
        # Test with single reference, with rotation
        rot_45 = Rotation.from_euler('z', 45, degrees=True).as_quat()
        rot_90 = Rotation.from_euler('z', 90, degrees=True).as_quat()
        
        demo_end = np.array([
            [0.0, 1.0, 0.0, *np.roll(rot_45, 1)],  # Target with 45-degree rotation
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # Reference
        ])
        
        start_poses = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # Target start
            [2.0, 0.0, 0.0, *np.roll(rot_90, 1)],   # Reference with 90-degree rotation
        ])
        
        traj = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, *np.roll(rot_45, 1)],
        ])
    
    return traj, demo_end, start_poses

def visualize_configurations(traj, demo_end_poses, start_poses, new_target, title="Test Configuration"):
    """Visualize test configurations"""
    fig = plt.figure(figsize=(15, 6))
    
    # Plot demonstration
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Demonstration Configuration")
    
    # Find target object index
    end_pos = traj[-1, :3]
    distances = np.linalg.norm(demo_end_poses[:, :3] - end_pos, axis=1)
    target_idx = np.argmin(distances)
    
    # Plot trajectory of target object
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'k--', alpha=0.5)
    
    # Plot and label start pose of target object
    plot_pose_axes(ax1, traj[0])
    ax1.text(traj[0, 0], traj[0, 1], traj[0, 2], 'Start', fontsize=10)
    
    # Plot demonstration end poses and label target end pose
    for i, pose in enumerate(demo_end_poses):
        plot_pose_axes(ax1, pose)
        # Add dotted line and label for target object
        if i == target_idx:
            ax1.plot([traj[0, 0], pose[0]], 
                    [traj[0, 1], pose[1]], 
                    [traj[0, 2], pose[2]], 
                    'k:', alpha=0.3)
            ax1.text(pose[0], pose[1], pose[2], 'End', fontsize=10)
    
    # Plot new configuration
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("New Configuration")
    
    # Plot start poses
    for i, pose in enumerate(start_poses):
        plot_pose_axes(ax2, pose)
        if i == 0:  # Target start pose
            ax2.text(pose[0], pose[1], pose[2], 'Start', fontsize=10)
    
    # Plot and label new target end pose
    plot_pose_axes(ax2, new_target)
    ax2.text(new_target[0], new_target[1], new_target[2], 'End', fontsize=10)
    
    # Connect start and end positions with lines
    ax2.plot([start_poses[0, 0], new_target[0]],
             [start_poses[0, 1], new_target[1]],
             [start_poses[0, 2], new_target[2]], 'k--', alpha=0.3)
    
    # Add legend and set view properties
    for ax in [ax1, ax2]:
        ax.plot([], [], 'r-', label='X-axis')
        ax.plot([], [], 'g-', label='Y-axis')
        ax.plot([], [], 'b-', label='Z-axis')
        ax.plot([], [], 'k--', label='Motion Path')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    # Make plot interactive
    def on_key_press(event):
        if event.key == 'r':
            for ax in [ax1, ax2]:
                ax.view_init(elev=30, azim=45)
            fig.canvas.draw()
    
    plt.ion()
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.text(0.5, 0.02, 'Use mouse to rotate/zoom. Press "r" to reset view.',
             ha='center', fontsize=9)
    plt.show(block=True)
    
    # Plot new configuration
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("New Configuration")
    
    # Plot start poses and new target
    for pose in start_poses:
        plot_pose_axes(ax2, pose)
    plot_pose_axes(ax2, new_target)
    
    # Connect start and end positions with lines
    ax2.plot([start_poses[0, 0], new_target[0]],
             [start_poses[0, 1], new_target[1]],
             [start_poses[0, 2], new_target[2]], 'k--', alpha=0.3)
    
    # Add legend and set view properties
    for ax in [ax1, ax2]:
        ax.plot([], [], 'r-', label='X-axis')
        ax.plot([], [], 'g-', label='Y-axis')
        ax.plot([], [], 'b-', label='Z-axis')
        ax.plot([], [], 'k--', label='Motion Path')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    # Make plot interactive
    def on_key_press(event):
        if event.key == 'r':
            for ax in [ax1, ax2]:
                ax.view_init(elev=30, azim=45)
            fig.canvas.draw()
    
    plt.ion()
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.text(0.5, 0.02, 'Use mouse to rotate/zoom. Press "r" to reset view.',
             ha='center', fontsize=9)
    plt.show(block=True)

def test_single_reference_cases():
    """Test single reference cases"""
    print("Testing single reference translation...")
    traj, demo_end, start_poses = generate_test_data("single_reference")
    new_target = get_new_end(traj, demo_end, start_poses, ignore_rotation=True)
    visualize_configurations(traj, demo_end, start_poses, new_target, "Translation Test")
    
    print("\nTesting single reference with rotation...")
    traj, demo_end, start_poses = generate_test_data("single_reference_rotation")
    new_target = get_new_end(traj, demo_end, start_poses, ignore_rotation=True)
    visualize_configurations(traj, demo_end, start_poses, new_target, "Rotation Test")

if __name__ == "__main__":
    test_single_reference_cases()