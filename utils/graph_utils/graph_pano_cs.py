import numpy as np
import torch


class Node(object):
    def __init__(self, idname, goal_num=6):
        self.is_start = False
        self.nodeid = idname

        self.rgb = None

        self.clip_feat = torch.zeros([12, 512]) # [rot num, feat dim]
        self.observed_feat = torch.zeros([12])
        self.vis_feat = torch.zeros([512])
        self.max_depth = torch.zeros([12])

        self.cm_score = torch.zeros([1])
        self.obj_value = torch.zeros([1])
        self.visited = torch.zeros([1])
        self.pos = torch.zeros([3])
        self.goal_cat = torch.zeros([goal_num])



        self.cm_name = None
        self.dist_to_objs = None

        self.goal_cm_info = None
        self.goal_cm_scores = None

        self.feat = torch.cat([self.vis_feat, self.cm_score, self.visited, self.pos, self.goal_cat], dim=0)
        self.rot = None

        self.draw = False

        self.children = []
        self.blocked_children_ids = []

        self.vis_pos = None
        self.invalid_pos = False



    def add_child(self, child):
        self.children.append(child)

    def set_to_start(self):
        self.is_start = True

    def update_feat(self):
        self.feat = torch.cat([self.vis_feat, torch.Tensor([self.cm_score]), self.visited, torch.Tensor(self.pos), self.goal_cat], dim=0)

    def update_clip_feat(self, feat, angle):
        feat = feat.cpu()
        self.clip_feat[angle] = (self.clip_feat[angle] * self.observed_feat[angle] + feat) / (self.observed_feat[angle] + 1)
        self.observed_feat[angle] += 1

    def update_max_depth(self, depth, angle):
        if depth > self.max_depth[angle]:
            self.max_depth[angle] = torch.Tensor([depth])

    def update_vis_feat(self):
        obs_feat = self.clip_feat[self.observed_feat == 1]
        self.vis_feat = torch.mean(obs_feat, dim=0)

    def update_cm_score(self, cm_score, cm_name=None, cm_info=None):
        self.cm_score = cm_score
        if cm_name is not None:
            self.cm_name = cm_name
        if cm_info is not None:
            self.goal_cm_info = cm_info

    def update_obj_value(self, obj_value):
        self.obj_value = obj_value

    def set_as_visted(self):
        self.visited = torch.ones([1])

    def update_pos(self, pos):
        self.pos = pos

    def update_goal_category(self, goal_cat):
        self.goal_cat = goal_cat

    def update_goal_cm_scores(self, goal_cm_scores, angle):
        goal_cm_scores = torch.Tensor(goal_cm_scores)
        self.goal_cm_scores[angle] = goal_cm_scores

    def __lt__(self, other):
        return int(self.nodeid) < int(other.nodeid)


class Edge(object):
    def __init__(self, node1, node2, weight, directional=False):
        self.nodes = (node1, node2)
        self.ids = (node1.nodeid, node2.nodeid)
        self.weight = weight
        self.directional = directional

        self.draw = False


class GraphMap(object):
    def __init__(self, args):
        self.args = args
        self.frame_width = args.pano_width
        self.frame_height = args.pano_height

        self.feat_dim = 512
        self.rot_num = int(360 / args.act_rot)

        self.nodes = set()
        self.node_by_id = {}
        self.poses = set()
        self.pose_to_id = {}
        self.edges = set()
        self.edge_ids = []
        self.edge_by_id = {}
        self.clustered = False
        self.clusterType = None
        self.graph_axes = None

        self.visited_node_ids = []
        self.candidate_node_ids = []

        # self.adj_mtx = None
        self.adj_mtx = np.zeros((1, 1))

        self.min_node_dist = self.args.edge_range / 2.
        self.max_edge_length = self.args.edge_range * 1.5

        self.goal_text_clip_feat = None
        self.goal_cm_info = None
        if hasattr(args, 'cm_type'):
            if args.cm_type == 'comet':
                self.goal_cm_length = 10
            elif args.cm_type == 'mp3d':
                self.goal_cm_length = 5
        else:
            self.goal_cm_length = None

    def set_axes(self):
        if len(self.nodes) == 0:
            self.graph_axes = {"x": [0, 1], "y": [0, 1]}
            return
        self.graph_axes = {}
        self.graph_axes["x"] = [
            min([i[0] for i in self.poses]) - 1,
            max([i[0] for i in self.poses]) + 1,
        ]
        self.graph_axes["y"] = [
            min([i[2] for i in self.poses]) - 1,
            max([i[2] for i in self.poses]) + 1,
        ]

    def total_nodes(self):
        return len(self.nodes)

    def get_node_by_id(self, nodeid):
        return self.node_by_id[nodeid]

    def get_node_by_pos(self, pos):
        pos = tuple([round(x, 4) for x in pos])
        return self.node_by_id[self.pose_to_id[pos]]

    def add_single_node(self, pos, min_node_dist=None):
        add_new_node = True
        if min_node_dist is None:
            min_node_dist = self.min_node_dist
        pos = tuple([round(x, 4) for x in pos])
        # if pos in self.poses:
        nearest_node_idx, nearest_node_dist = self.get_nearest_node(pos, for_localization=False)
        if nearest_node_dist < min_node_dist:
            node = self.node_by_id[nearest_node_idx]
            add_new_node = False
        else:
            nodeid = str(self.total_nodes())
            node = Node(nodeid)
            node.update_pos(pos)

            self.nodes.add(node)
            self.node_by_id[nodeid] = node
            self.poses.add(pos)
            self.pose_to_id[pos] = nodeid
            self.candidate_node_ids.append(nodeid)
            self.expand_adj_mtx()

            if self.goal_cm_length is not None:
                node.goal_cm_scores = torch.zeros([12, self.goal_cm_length])

        return node, add_new_node

    def get_nearest_node(self, pos, except_node_id=None, for_localization=True):
        if len(self.poses) == 0:
            return None, 999
        poses = list(self.poses)
        if for_localization:
            if len(self.adj_mtx) >1:
                isolated_node_ids = np.where(np.sum(self.adj_mtx, axis=0) == 0)[0]
                for id in isolated_node_ids:
                    poses.remove(self.node_by_id[str(id)].pos)

        if not except_node_id is None:
            for id in except_node_id:
                if self.node_by_id[id].pos in poses:
                    poses.remove(self.node_by_id[id].pos)
        poses = np.array(poses)
        pos_diff = np.array([np.linalg.norm(np.array([pos[0], pos[2]]) - np.array([p[0], p[2]])) for p in poses])
        pos_idx = np.argmin(pos_diff)
        node_idx = self.pose_to_id[tuple(poses[pos_idx])]
        return str(node_idx), pos_diff[pos_idx]

    def check_node_exist(self, pos):
        if len(self.poses) == 0:
            return False
        pos_diff = np.array([np.linalg.norm(np.array([pos[0], pos[2]]) - np.array([p[0], p[2]])) for p in self.poses])
        if np.min(pos_diff) < self.min_node_dist:
            return True
        else:
            return False

    def update_node_is_start(self, node):
        node.is_start = True



    def update_node_clip_feat(self, node, feat, angle):
        """Update the clip feature of a node"""
        node.update_clip_feat(feat, angle)

    def update_node_max_depth(self, node, depth, angle):
        """Update the clip feature of a node"""
        node.update_max_depth(depth, angle)

    def update_node_vis_feat(self, node):
        """Update the visual feature of a node"""
        node.update_vis_feat()

    def update_node_cm_score(self, node, cm_score, cm_name=None, cm_info=None):
        """Update the score of a node"""
        node.update_cm_score(cm_score, cm_name, cm_info)

    def update_node_obj_value(self, node, obj_value):
        """Update the score of a node"""
        node.update_obj_value(obj_value)


    def update_node_pos(self, node, pos):
        """Update the position of a node"""
        pos = tuple([round(x, 4) for x in pos])
        prev_pos = node.pos
        node.update_pos(pos)
        self.poses.remove(prev_pos)
        self.poses.add(pos)
        self.pose_to_id.pop(prev_pos)
        self.pose_to_id[pos] = node.nodeid

    def update_node_rot(self, node, rot):
        """Update the rotation of a node"""
        node.rot = rot

    def update_node_visited(self, node):
        node.set_as_visted()
        if not node.nodeid in self.visited_node_ids:
            self.visited_node_ids.append(node.nodeid)
        if node.nodeid in self.candidate_node_ids:
            self.candidate_node_ids.remove(node.nodeid)
        # self.visited_node_ids.append(node.nodeid)
        # self.candidate_node_ids.remove(node.nodeid)

    def update_node_goal_category(self, node, goal_category):
        node.update_goal_category(goal_category)


    def update_node_obs(self, node, rgb):
        """Update the observation of a node"""
        node.rgb = rgb


    def update_node_feat(self, node, vis_feat=None, cm_score=None, visited=None, pos=None):
        """Update the feature of a node"""
        if vis_feat is not None:
            node.update_vis_feat(vis_feat)
        if cm_score is not None:
            node.update_cm_score(cm_score)
        if visited is not None:
            node.set_as_visted()
        if pos is not None:
            node.update_pos(pos)
        node.update_feat()


    def update_node_dist_to_objs(self, node, dist_to_objs):
        """Add the ged distance to the node"""
        node.dist_to_objs = dist_to_objs

    def add_edge(self, node1, node2):
        if node1.nodeid == node2.nodeid:
            return
        if node1 not in self.nodes or node2 not in self.nodes:
            print("error one or more of the nodes not in the graph")
            return
        if (node1.nodeid, node2.nodeid) in self.edge_ids or (node2.nodeid, node1.nodeid) in self.edge_ids:
            return

        if node1.nodeid in node2.blocked_children_ids or node2.nodeid in node1.blocked_children_ids:
            return

        if node1.invalid_pos or node2.invalid_pos:
            return

        distance = np.linalg.norm(np.asarray(node1.pos) - np.asarray(node2.pos))
        # if distance > self.max_edge_length:
        #     return
        edge = Edge(node1, node2, distance)
        self.edges.add(edge)
        self.edge_ids.append((node1.nodeid, node2.nodeid))
        self.edge_ids.append((node2.nodeid, node1.nodeid))
        node1.children.append({'node': node2, 'nodeid': node2.nodeid})
        node2.children.append({'node': node1, 'nodeid': node1.nodeid})

        self.edge_by_id[(node1.nodeid, node2.nodeid)] = edge
        self.edge_by_id[(node2.nodeid, node1.nodeid)] = edge

        self.update_adj_mtx(node1, node2, distance)

    def delete_edge(self, node1, node2):
        edge_ids = [edge for edge in self.edge_ids if edge == (node1.nodeid, node2.nodeid) or edge == (node2.nodeid, node1.nodeid)]
        if len(edge_ids) == 0:
            return
        self.edge_ids.remove(edge_ids[0])
        self.edge_ids.remove(edge_ids[1])
        remove_list = []
        for edge in self.edges:
            if edge.ids == edge_ids[0] or edge.ids == edge_ids[1]:
                remove_list.append(edge)
                break
        for edge in remove_list:
            self.edges.remove(edge)

        for child in node1.children:
            if child['nodeid'] == node2.nodeid:
                node1.children.remove(child)
                break
        for child in node2.children:
            if child['nodeid'] == node1.nodeid:
                node2.children.remove(child)
                break

        self.adj_mtx[int(node1.nodeid), int(node2.nodeid)] = 0
        self.adj_mtx[int(node2.nodeid), int(node1.nodeid)] = 0

        self.edge_by_id.pop((node1.nodeid, node2.nodeid))
        self.edge_by_id.pop((node2.nodeid, node1.nodeid))

        node1.blocked_children_ids.append(node2.nodeid)
        node2.blocked_children_ids.append(node1.nodeid)

    def delete_invalid_node(self, node):
        children = [child['node'] for child in node.children]
        for child in children:
            self.delete_edge(node, child)
        node.invalid_pos = True

    def merge_nodes(self, node1, node2):
        """Merge two nodes in the graph"""
        if node1 == node2:
            return
        if node1 in node2.children or node2 in node1.children:
            return
        for child in node2.children:
            if child == node1:
                continue
            self.add_edge(child, node1)
            child.children.remove(node2)
        self.nodes.remove(node2)
        self.node_by_id.pop(node2.nodeid)
        self.poses.remove(node2.pos)
        self.pose_to_id.pop(node2.pos)
        self.edges = set([e for e in self.edges if node2 not in e.nodes])

    def expand_adj_mtx(self):
        adj_mtx_size = len(self.nodes)
        if adj_mtx_size > self.adj_mtx.shape[0]:
            adj_mtx = np.zeros((adj_mtx_size, adj_mtx_size))
            adj_mtx[:self.adj_mtx.shape[0], :self.adj_mtx.shape[1]] = self.adj_mtx
            self.adj_mtx = adj_mtx


    def update_adj_mtx(self, node1, node2, distance):
        """Update the adjacency matrix"""
        self.expand_adj_mtx()
        self.adj_mtx[int(node1.nodeid)][int(node2.nodeid)] = distance
        self.adj_mtx[int(node2.nodeid)][int(node1.nodeid)] = distance


    def add_cand_nodes(self, node, cand_nodes, dirc_feats, obj_info_list):
        node.cand_nodes = cand_nodes
        for i, cand in enumerate(cand_nodes['free_cand_angle']):
            if cand == 0:
                continue
            pos = tuple([round(x,4) for x in cand_nodes['position'][i]])
            if pos in self.poses:
                cur_cand_node = self.node_by_id[self.pose_to_id[pos]]
                if cur_cand_node.visited:
                    self.add_edge(node, cur_cand_node)
                else:
                    self.add_edge(node, cur_cand_node, directional=True)
                continue

            nodeid = str(self.total_nodes())
            cur_cand_node = Node(nodeid, pos)
            cur_cand_node.visited = False
            feat = torch.zeros([self.rot_num, self.feat_dim])
            feat[0, :] = dirc_feats[i]
            cur_cand_node.update_feat(feat)
            cur_cand_node.room = cand_nodes['room'][i]
            cur_cand_node.ged_dist = cand_nodes['dist_to_objs'][i]
            cur_cand_node.update_objects(obj_info_list[i])
            self.nodes.add(cur_cand_node)
            self.node_by_id[nodeid] = cur_cand_node
            self.poses.add(pos)
            self.pose_to_id[pos] = nodeid
            self.add_edge(node, cur_cand_node, directional=True)


def affinity_cluster(G):
    # print('afinity propagation on G')

    """get feats for each node and cluster using sklearn kmeans"""
    X = []
    for i in range(len(G.nodes)):
        node = G.node_by_id[str(i)]
        feat = node.feat[0].detach().numpy()
        pos = node.pos
        full_feat = np.concatenate((feat, pos), axis=0)
        X.append(full_feat)
    from sklearn.cluster import AffinityPropagation

    clustering = AffinityPropagation(random_state=5).fit(X)
    labels = clustering.labels_

    """create new graph"""
    clusteredGraph = GraphMap(params=G.params)
    if G.graph_axes is None:
        G.set_axes()
    clusteredGraph.graph_axes = G.graph_axes

    """create the centriod nodes and add them to the new graph"""
    node_clusters = {i: [] for i in range(max(labels) + 1)}
    for i in range(len(G.nodes)):
        node = G.node_by_id[str(i)]
        node_clusters[labels[i]].append(node)

    clusteredGraph = centriod_from_cluster(clusteredGraph, node_clusters)
    clusteredGraph = add_edges_to_new_graph(G, clusteredGraph, labels)

    clusteredGraph.clustered = True
    clusteredGraph.clusterType = "affinity"
    return clusteredGraph


def add_edges_to_new_graph(G, clusteredGraph, labels):
    edgeList = []
    for edge in G.edges:
        id1 = str(int(labels[int(edge.ids[0])]))
        id2 = str(int(labels[int(edge.ids[1])]))
        if id1 == id2:
            continue
        if {id1, id2} in edgeList:
            continue
        edgeList.append({id1, id2})
        node1 = clusteredGraph.node_by_id[id1]
        node2 = clusteredGraph.node_by_id[id2]
        clusteredGraph.add_edge(node1, node2)

    for i in range(len(clusteredGraph.nodes) - 2):
        id1 = str(i)
        id2 = str(i + 1)
        if {id1, id2} in edgeList:
            continue
        edgeList.append({id1, id2})
        node1 = clusteredGraph.node_by_id[id1]
        node2 = clusteredGraph.node_by_id[id2]
        clusteredGraph.add_edge(node1, node2)

    return clusteredGraph


def centriod_from_cluster(clusteredGraph, node_clusters):
    mean = False
    for label in range(len(node_clusters)):
        cluster = node_clusters[label]
        middle = round(len(cluster) * 1.0 / 2)
        # mean
        if mean:
            centriod_pos = np.mean([x.pos for x in cluster], axis=0)
            centriod_rot = cluster[middle].rot
            centriod_feat = [x for n in cluster for x in n.feat]
            centriod_feat = torch.mean(torch.stack(centriod_feat), dim=0)
        # center
        else:
            centriod_pos = cluster[middle].pos
            centriod_rot = cluster[middle].rot
            centriod_feat = cluster[middle].feat[0]

        added_node = clusteredGraph.add_single_node(
            centriod_pos, centriod_rot, centriod_feat
        )
        has_start = [x.is_start for x in cluster]
        if True in has_start:
            added_node.set_to_start()

    return clusteredGraph