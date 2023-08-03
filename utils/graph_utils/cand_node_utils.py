import numpy as np
# import matplotlib.pyplot as plt

def get_cand_nodes(pos, edge_range):
    cand_angles = [0, -30, 30]
    rot_axis = np.array([0, 1, 0])

    node_poses = []
    for angle in cand_angles:
        rot_vec = np.radians(-angle) * rot_axis
        unit_vec = -np.array([np.sin(rot_vec[1]), 0, np.cos(rot_vec[1])])
        cand_pos = pos + unit_vec * edge_range
        node_poses.append(cand_pos)
    return node_poses


def get_range_cand_nodes(max_range, edge_range):
    pos = np.zeros(3)
    edge_range = edge_range

    node_poses = get_cand_nodes(pos, edge_range)
    total_node_poses = node_poses.copy()
    next_target_poses = node_poses.copy()

    for j in range(max_range-1):
        target_poses = next_target_poses
        next_target_poses = []
        for node_pose in target_poses:
            temp_node_poses = get_cand_nodes(node_pose, edge_range)
            for temp_node_pose in temp_node_poses:
                if min(np.linalg.norm(np.array(total_node_poses) - temp_node_pose, axis=1)) > 0.5 * edge_range:
                    total_node_poses.append(temp_node_pose)
                    next_target_poses.append(temp_node_pose)

    total_node_rots = []
    for pose in total_node_poses:
        total_node_rots.append(np.array([0, -np.arctan2(pose[0], -pose[2]), 0]))

    return {
        'poses': np.array(total_node_poses),
        'rots': np.array(total_node_rots)
    }