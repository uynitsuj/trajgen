import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from traj_interp import get_new_end, get_circle_parameters

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

def generate_sample_data():
    """Generate sample data for testing"""
    # Create a simple demonstration with 3 objects
    # Object 1 will be our target object that moves between two other objects
    demo_objects_start = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Target object
        [-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Reference object 1
        [1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # Reference object 2
    ])
    
    # Add some rotation to make the visualization more interesting
    rot_quat = Rotation.from_euler('xyz', [0, 0, 45], degrees=True).as_quat()  # xyzw
    end_orientation = np.array([rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]])  # wxyz
    
    demo_objects_end = np.array([
        [0.0, -1.0, 1.0, *end_orientation],    # Target object ends up between the two with rotation
        [-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Reference object 1 stays put
        [1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0],   # Reference object 2 stays put
    ])
    
    # Create a simple trajectory for the target object
    t = np.linspace(0, 1, 10)
    x = 0.0 * np.ones_like(t)
    y = t
    z = 0.0 * np.ones_like(t)
    
    # Interpolate orientation along the trajectory
    rots = []
    for alpha in t:
        rot_quat = Rotation.from_euler('xyz', [0, 0, 45 * alpha], degrees=True).as_quat()
        rots.append([rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]])  # wxyz
    
    traj = np.column_stack((x, y, z, np.array(rots)))
    
    return traj, demo_objects_start, demo_objects_end

def visualize_configurations(demo_start, demo_end, new_start, new_target):
    """
    Visualize both the demonstration and new configurations with coordinate frames
    
    Args:
        demo_start: Demonstration starting poses
        demo_end: Demonstration ending poses
        new_start: New starting poses
        new_target: New target pose
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Plot demonstration
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Demonstration Configuration")
    
    # Plot start poses
    for pose in demo_start:
        plot_pose_axes(ax1, pose)
    
    # Plot end poses
    for pose in demo_end:
        plot_pose_axes(ax1, pose)
    
    # Connect start and end positions with lines
    for i in range(len(demo_start)):
        ax1.plot([demo_start[i, 0], demo_end[i, 0]],
                [demo_start[i, 1], demo_end[i, 1]],
                [demo_start[i, 2], demo_end[i, 2]], 'k--', alpha=0.3)
    
    # Plot new configuration
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("New Configuration")
    
    # Plot start poses
    for pose in new_start:
        plot_pose_axes(ax2, pose)
    
    # Combine new_target with other objects' positions for end poses
    new_end = np.vstack([new_target, new_start[1:]])
    for pose in new_end:
        plot_pose_axes(ax2, pose)
    
    # Connect start and end positions with lines
    for i in range(len(new_start)):
        ax2.plot([new_start[i, 0], new_end[i, 0]],
                [new_start[i, 1], new_end[i, 1]],
                [new_start[i, 2], new_end[i, 2]], 'k--', alpha=0.3)
    
    # Add legend to both plots
    for ax in [ax1, ax2]:
        ax.plot([], [], 'r-', label='X-axis')
        ax.plot([], [], 'g-', label='Y-axis')
        ax.plot([], [], 'b-', label='Z-axis')
        ax.plot([], [], 'k--', label='Motion Path')
    
    # Set equal aspect ratio and labels
    for ax in [ax1, ax2]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_box_aspect([1,1,1])
        
        # Set consistent axis limits
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        
        # Set view angle
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    # Make the plot interactive
    def on_key_press(event):
        if event.key == 'r':  # Reset view
            for ax in [ax1, ax2]:
                ax.view_init(elev=30, azim=45)
            fig.canvas.draw()
    
    # Enable interactive mode
    plt.ion()
    
    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Add instructions text
    fig.text(0.5, 0.02, 'Use mouse to rotate/zoom. Press "r" to reset view.', 
             ha='center', fontsize=9)
    
    # Show the plot
    plt.show(block=True)  # block=True ensures the window stays open

def test_target_generation():
    """Test the target generation with different configurations"""
    # Generate sample data for demonstration
    traj, demo_start, demo_end = generate_sample_data()
    
    # Create a new starting configuration by rotating and scaling the reference objects
    theta = np.pi / 4  # 45-degree rotation
    scale = 1.5
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    new_start = demo_start.copy()
    # Rotate and scale the reference objects (objects 1 and 2)
    new_start[1:, :3] = scale * (rotation_matrix @ demo_start[1:, :3].T).T
    
    # Generate new target position
    new_target = get_new_end(traj, demo_end, new_start)
    
    # Visualize results
    visualize_configurations(demo_start, demo_end, new_start, new_target)
    
    return new_target


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def plot_circle_plane(ax, center, normal, radius, color='cyan'):
    """Plot the plane containing the intersection circle"""
    # Create a circular grid of points
    r = np.linspace(0, radius*1.5, 10)  # Extend a bit beyond the circle
    theta = np.linspace(0, 2*np.pi, 20)
    r, theta = np.meshgrid(r, theta)
    
    # Create basis vectors for the plane
    if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
    
    # Calculate plane points
    points = np.zeros((3, r.shape[0], r.shape[1]))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            point = center + r[i,j] * (u * np.cos(theta[i,j]) + v * np.sin(theta[i,j]))
            points[:, i, j] = point
    
    # Plot the plane
    ax.plot_surface(points[0], points[1], points[2], alpha=0.2, color=color)

def plot_plane(ax, normal, d, center, color='yellow', size=2):
    """Plot a plane given its normal and distance from origin"""
    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(center[0]-size, center[0]+size, 10),
                        np.linspace(center[1]-size, center[1]+size, 10))
    
    if abs(normal[2]) > 1e-6:
        zz = center[2] + (d - normal[0]*(xx-center[0]) - normal[1]*(yy-center[1])) / normal[2]
        ax.plot_surface(xx, yy, zz, alpha=0.2, color=color)
    elif abs(normal[1]) > 1e-6:
        zz = np.meshgrid(np.linspace(center[2]-size, center[2]+size, 10),
                        np.linspace(center[1]-size, center[1]+size, 10))[0]
        yy = center[1] + (d - normal[0]*(xx-center[0]) - normal[2]*(zz-center[2])) / normal[1]
        ax.plot_surface(xx, yy, zz, alpha=0.2, color=color)
    elif abs(normal[0]) > 1e-6:
        zz = np.meshgrid(np.linspace(center[2]-size, center[2]+size, 10),
                        np.linspace(center[1]-size, center[1]+size, 10))[0]
        xx = center[0] + (d - normal[1]*(yy-center[1]) - normal[2]*(zz-center[2])) / normal[0]
        ax.plot_surface(xx, yy, zz, alpha=0.2, color=color)

def visualize_geometry(ax, A, B, d1, d2, ABS_normal, scaled_d, C, S, center=None, circle_normal=None, radius=None):
    """Visualize geometric constructs for debugging"""
    # Plot reference points
    ax.scatter(*A, c='b', marker='o', label='A', s=100)
    ax.scatter(*B, c='b', marker='o', label='B', s=100)
    ax.scatter(*C, c='g', marker='o', label='C', s=100)
    ax.scatter(*S, c='r', marker='o', label='S', s=100)
    
    # Plot line AB
    ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], 'k--', label='AB line')
    
    # Plot spheres (wireframe)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    # Sphere around A
    x = d1 * np.outer(np.cos(u), np.sin(v)) + A[0]
    y = d1 * np.outer(np.sin(u), np.sin(v)) + A[1]
    z = d1 * np.outer(np.ones_like(u), np.cos(v)) + A[2]
    ax.plot_wireframe(x, y, z, color='r', alpha=0.1)
    
    # Sphere around B
    x = d2 * np.outer(np.cos(u), np.sin(v)) + B[0]
    y = d2 * np.outer(np.sin(u), np.sin(v)) + B[1]
    z = d2 * np.outer(np.ones_like(u), np.cos(v)) + B[2]
    ax.plot_wireframe(x, y, z, color='b', alpha=0.1)
    
    # Plot ABS plane
    if not np.allclose(ABS_normal, 0):
        plot_plane(ax, ABS_normal, scaled_d, A, color='g')
    
    # Plot intersection circle construction if provided
    if center is not None and circle_normal is not None and radius is not None:
        # Plot center point
        ax.scatter(*center, c='m', marker='*', s=200, label='Circle Center')
        
        # Plot line connecting sphere centers with extension
        line_length = np.linalg.norm(B - A)
        line_dir = (B - A) / line_length
        line_start = A - line_dir * line_length * 0.2
        line_end = B + line_dir * line_length * 0.2
        ax.plot([line_start[0], line_end[0]], 
                [line_start[1], line_end[1]], 
                [line_start[2], line_end[2]], 
                'k-', alpha=0.5, label='Center Line')
        
        # Plot distance from A to circle center
        ax.plot([A[0], center[0]], [A[1], center[1]], [A[2], center[2]], 
                'g--', alpha=0.5, label='Center Distance')
        
        # Plot the intersection circle
        theta = np.linspace(0, 2*np.pi, 100)
        if np.allclose(circle_normal, [0, 0, 1]) or np.allclose(circle_normal, [0, 0, -1]):
            u = np.array([1, 0, 0])
            v = np.array([0, 1, 0])
        else:
            u = np.cross(circle_normal, [0, 0, 1])
            u = u / np.linalg.norm(u)
            v = np.cross(circle_normal, u)
            v = v / np.linalg.norm(v)
        
        circle_points = np.zeros((3, len(theta)))
        for i, t in enumerate(theta):
            circle_points[:, i] = center + radius * (u * np.cos(t) + v * np.sin(t))
        ax.plot(circle_points[0], circle_points[1], circle_points[2], 
                'r-', linewidth=2, label='Intersection Circle')
        
        # Plot radius vector
        radius_end = center + radius * u
        ax.plot([center[0], radius_end[0]], 
                [center[1], radius_end[1]], 
                [center[2], radius_end[2]], 
                'r--', alpha=0.5, label='Circle Radius')
    
    # Plot CS vector
    CS = S - C
    ax.quiver(C[0], C[1], C[2], CS[0], CS[1], CS[2], 
             color='m', alpha=0.5, label='CS Vector')
    return ax

def plot_circle(ax, center, radius, normal):
    """Plot a circle in 3D space"""
    # Create basis vectors
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
    elif np.allclose(normal, [0, 0, -1]):
        u = np.array([1, 0, 0])
        v = np.array([0, -1, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        if np.allclose(u, 0):
            u = np.array([1, 0, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
    
    # Generate points around the circle
    theta = np.linspace(0, 2*np.pi, 100)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Calculate points
    points = np.zeros((3, 100))
    points[0, :] = center[0] + radius * (u[0] * cos_theta + v[0] * sin_theta)
    points[1, :] = center[1] + radius * (u[1] * cos_theta + v[1] * sin_theta)
    points[2, :] = center[2] + radius * (u[2] * cos_theta + v[2] * sin_theta)
    
    ax.plot(points[0], points[1], points[2], 'r-', label='Intersection Circle')

def debug_geometry():
    """Debug geometric calculations with visualization"""
    # Generate sample data
    traj = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Start
        [0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # End
    ])
    
    demo_end_poses = np.array([
        [0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # Target
        [-1.5, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Reference 1
        [1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Reference 2
    ])
    
    # Extract points
    end_pos = traj[-1, :3]
    distances = np.linalg.norm(demo_end_poses[:, :3] - end_pos, axis=1)
    object_idx = np.argmin(distances)
    ref_idx = [i for i in range(len(demo_end_poses)) if i != object_idx]
    
    S, E = traj[0, :3], traj[-1, :3]
    A, B = demo_end_poses[ref_idx[0], :3], demo_end_poses[ref_idx[1], :3]
    
    # Calculate midpoint C
    d_AS = np.linalg.norm(S - A)
    d_BS = np.linalg.norm(S - B)
    C = A + (d_AS / (d_AS + d_BS)) * (B - A)
    
    # Calculate sphere radii
    d1 = np.linalg.norm(E - A)
    d2 = np.linalg.norm(E - B)
    
    # Calculate plane normal
    CS = S - C
    old_AB = B - A
    ABS_normal = np.cross(old_AB, CS)
    if not np.allclose(ABS_normal, 0):
        ABS_normal = ABS_normal / np.linalg.norm(ABS_normal)
    
    # Calculate signed distance
    d = np.dot(E - A, ABS_normal)
    
    # Print debug info for original configuration
    print("Original configuration:")
    print(f"Points A={A}, B={B}, S={S}, E={E}, C={C}")
    print(f"Distances: d1={d1:.3f}, d2={d2:.3f}")
    print(f"Plane normal: {ABS_normal}")
    print(f"Plane distance: {d:.3f}")
    print(f"CS vector: {CS}")
    
    # Create new configuration with rotation and scale
    theta = np.pi / 4  # 45-degree rotation
    scale = 1.5
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Transform points
    A_new = scale * (R @ A)
    B_new = scale * (R @ B)
    new_S = scale * (R @ S)
    
    # Calculate new parameters
    d1_new, d2_new = d1 * scale, d2 * scale
    C_new = A_new + (d_AS / (d_AS + d_BS)) * (B_new - A_new)
    new_CS = new_S - C_new
    new_AB = B_new - A_new
    new_ABS_normal = np.cross(new_AB, new_CS)
    if not np.allclose(new_ABS_normal, 0):
        new_ABS_normal = new_ABS_normal / np.linalg.norm(new_ABS_normal)
    
    scaled_d = d * scale
    
    # Print debug info for new configuration
    print("\nNew configuration:")
    print(f"Points A'={A_new}, B'={B_new}, S'={new_S}, C'={C_new}")
    print(f"Scaled distances: d1_new={d1_new:.3f}, d2_new={d2_new:.3f}")
    print(f"New plane normal: {new_ABS_normal}")
    print(f"Scaled plane distance: {scaled_d:.3f}")
    print(f"New CS vector: {new_CS}")
    
    # Visualize original configuration
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original Configuration")
    ax1 = visualize_geometry(ax1, A, B, d1, d2, ABS_normal, d, C, S)
    ax1.scatter(*E, c='r', marker='^', label='E')
    
    # Get circle parameters for new configuration
    center, radius, circle_normal = get_circle_parameters(A_new, B_new, d1_new, d2_new)
    
    # Visualize new configuration
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("New Configuration")
    ax2 = visualize_geometry(ax2, A_new, B_new, d1_new, d2_new, new_ABS_normal, 
                           scaled_d, C_new, new_S, center, circle_normal, radius)
    
    # Set consistent view
    for ax in [ax1, ax2]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    debug_geometry()

# if __name__ == "__main__":
#     # Run the test
#     new_target = test_target_generation()
#     print("Generated target pose:", new_target)