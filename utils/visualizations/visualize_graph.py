import numpy as np
import quaternion

import cv2
import matplotlib.pyplot as plt
from .maps import to_grid, TopdownView
from ..graph_utils.graph_pano import GraphMap
from ..obj_category_info import obj_names_det as obj_names, goal_obj_names, room_names



def get_vis_grid_pose(pose, map_size, bounds):
    if len(pose) == 3:
        grid_y, grid_x = to_grid(pose[2], pose[0], map_size, bounds=bounds)
    elif len(pose) == 2:
        grid_y, grid_x = to_grid(pose[1], pose[0], map_size, bounds=bounds)
    return (grid_x, grid_y)


def node_value_by_obj_dist(dist, max_dist=15.0):
    return max(1 - dist / max_dist, 0)


def vis_topdown_graph_map(vis_map, graph_map, vis_obj_score=None):
    # vis_obj_score == target object class index

    # vis_map = (np.tile(np.expand_dims(self.map, -1), 3) / 2. * 255).astype(np.uint8)
    node_list = list(graph_map.node_by_id.values())

    for edge in list(graph_map.edges):
        node_grid1 = get_vis_grid_pose(edge.nodes[0].pos, map_size, sim)
        node_grid2 = get_vis_grid_pose(edge.nodes[1].pos, map_size, sim)
        vis_map = cv2.line(vis_map, node_grid1, node_grid2, (0, 64, 64), 5)

    for node in node_list:
        node_grid = self.get_vis_grid_pose(node.pos)
        if vis_obj_score is not None:
            color = (np.array((0, 255, 0)) * self.node_value_by_obj_dist(node.dist_to_objs[vis_obj_score])).astype(int)
            color = tuple([color[i].item() for i in range(3)])
            cand_color = (np.array((0, 0, 255)) * self.node_value_by_obj_dist(node.dist_to_objs[vis_obj_score])).astype(
                int)
            cand_color = tuple([cand_color[i].item() for i in range(3)])
        else:
            color = (0, 255, 0)
            cand_color = (0, 0, 255)
        if node.visited:
            vis_map = cv2.circle(vis_map, node_grid, 10, color, -1)
        else:
            vis_map = cv2.circle(vis_map, node_grid, 10, cand_color, -1)

    return vis_map


def get_bbox_from_pos_size(self, position, size):
    bbox = np.array([
        [position[0] - size[0] / 2, position[2] - size[2] / 2],
        [position[0] + size[0] / 2, position[2] + size[2] / 2]
    ])
    grid_bbox = np.array([
        self.get_vis_grid_pose(bbox[0]),
        self.get_vis_grid_pose(bbox[1])
    ])
    return grid_bbox


def vis_topdown_obj_map(self, vis_map, obj_category):
    for obj in self.env_obj_info:
        agent_level = self.check_position2level(self.scene_height)
        if obj['category'] == obj_category and agent_level == obj['level']:
            grid_bbox = self.get_bbox_from_pos_size(obj['position'], obj['sizes'])
            grid_bbox = grid_bbox.tolist()
            shapes = np.zeros_like(vis_map, np.uint8)
            cv2.rectangle(shapes, tuple(grid_bbox[0]), tuple(grid_bbox[1]), (0, 0, 128), -1)
            alpha = 0.3
            mask = shapes.astype(bool)
            vis_map[mask] = cv2.addWeighted(vis_map, alpha, shapes, 1 - alpha, 0)[mask]

    return vis_map


def vis_topdown_goal_obj_map(self, vis_map, goal_obj_category):
    for obj in self.env_goal_obj_info:
        agent_level = self.check_position2level(self.scene_height)
        if obj['category'] == goal_obj_category and agent_level == obj['level']:
            grid_bbox = self.get_bbox_from_pos_size(obj['position'], obj['sizes'])
            grid_bbox = grid_bbox.tolist()
            shapes = np.zeros_like(vis_map, np.uint8)
            cv2.rectangle(shapes, tuple(grid_bbox[0]), tuple(grid_bbox[1]), (255, 0, 0), -1)
            alpha = 0.3
            mask = shapes.astype(bool)
            vis_map[mask] = cv2.addWeighted(vis_map, alpha, shapes, 1 - alpha, 0)[mask]
            # vis_map = cv2.rectangle(vis_map, tuple(grid_bbox[0]), tuple(grid_bbox[1]), (255, 0, 0), -1)
    return vis_map


def vis_topdown_map_with_captions(self, graph_map, vis_goal_obj_score=None, vis_obj=None, save_dir=None):
    vis_map = self.map.copy()
    if vis_obj is not None:
        vis_map = self.vis_topdown_obj_map(vis_map, vis_obj)
    if vis_goal_obj_score is not None:
        vis_map = self.vis_topdown_goal_obj_map(vis_map, vis_goal_obj_score)
    vis_map = self.vis_topdown_graph_map(vis_map, graph_map, vis_goal_obj_score)

    plt.imshow(vis_map)
    plt.axis('off')
    plt.tight_layout()
    plt.title(f'Target object goal : {goal_obj_names[vis_goal_obj_score]}\n', wrap=True, horizontalalignment='center',
              fontsize=12)
    # plt.figtext(0.5, 0.1, f'Target object goal : {goal_obj_names[vis_goal_obj_score]}\n', wrap=True, horizontalalignment='center', fontsize=12)
    if vis_obj is not None:
        plt.figtext(0.5, 0.05, f'Object position : {obj_names[vis_obj]}\n', wrap=True, horizontalalignment='center',
                    fontsize=8)
    if save_dir is not None:
        plt.savefig(save_dir)
    else:
        plt.show()
    plt.close()


def vis_pos_on_topdown_map(self, pos):
    vis_map = (np.tile(np.expand_dims(self.map, -1), 3) / 2. * 255).astype(np.uint8)
    node_grid = self.get_vis_grid_pose(pos)
    vis_map = cv2.circle(vis_map, node_grid, 4, (0, 255, 0), -1)
    return vis_map


def calculate_navmesh(self):
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_success = self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings, include_static_objects=True)
    print("navmesh_success ", navmesh_success)
    self.tdv = TopdownView(self._sim, 'mp3d', data_dir=self.args.floorplan_data_dir)
    self.scene_height = self._sim.agents[0].state.position[1]
    self.tdv.draw_top_down_map(height=self.scene_height)
    self.curr_rot = self.tdv.get_polar_angle(self.init_rotation)


def get_topdown_floorplan(self):
    if abs(self._sim.agents[0].state.position[1] - self.scene_height) > 0.5:
        self.scene_height = self._sim.agents[0].state.position[1]
        self.tdv.draw_top_down_map(height=self.scene_height)
    # tdv = self.tdv.draw_agent()
    # if len(self.map_paths) > 0:
    #     tdv = self.tdv.draw_paths(self.map_paths, tdv)
    return self.tdv.rgb_top_down_map




def traj_to_graph(args, traj_data):
    graph_map = GraphMap(args)

    pano_size = (int(args.pano_height / 2), int(args.pano_width / 2))
    pano_height = pano_size[0]
    pano_width = pano_size[1]
    dirc_size = (int(args.img_height / 2), int(args.img_width / 2))


    node_idx = traj_data['node_idx']


    node_list = []
    for i, node in enumerate(node_idx):
        pred_classes, scores, obj_info, masks, boxes = detector_module.predicted_img(rgb_list[i], show=False)
        dirc_feats = get_feat(rgbd_list[i], encoder, args.use_depth)
        obj_info_list = get_dirc_det(rgb_list[i])
        cur_node = graph_map.add_single_node(seq_data['position'][node],
                                             dirc_feats,
                                             obj_info,
                                             seq_data['room'][node]
                                             )
        graph_map.add_node_ged_dist(cur_node, seq_data['dist_to_objs'][node])
        graph_map.add_cand_nodes(cur_node, seq_data['cand_nodes'][node], dirc_feats, obj_info_list)

        node_list.append(cur_node)

    for i, n in enumerate(node_list[:-1]):
        graph_map.add_edge(node_list[i], node_list[i + 1])
        graph_map.add_edge(node_list[i + 1], node_list[i])


