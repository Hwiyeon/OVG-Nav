
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import habitat_sim
import habitat_sim.agent
import habitat_sim.bindings as hsim
import numpy as np
import magnum as mn
import pickle
import os


default_sim_settings = {
    # settings shared by example.py and benchmark.py
    "max_frames": 500,
    "width": 640,
    "height": 480,
    "default_agent": 0,
    "sensor_height": 1.5,
    "hfov": 90,
    "color_sensor": True,  # RGB sensor (default: ON)
    "semantic_sensor": False,  # semantic sensor (default: OFF)
    "depth_sensor": False,  # depth sensor (default: OFF)
    "ortho_rgba_sensor": False,  # Orthographic RGB sensor (default: OFF)
    "ortho_depth_sensor": False,  # Orthographic depth sensor (default: OFF)
    "ortho_semantic_sensor": False,  # Orthographic semantic sensor (default: OFF)
    "fisheye_rgba_sensor": False,
    "fisheye_depth_sensor": False,
    "fisheye_semantic_sensor": False,
    "equirect_rgba_sensor": False,
    "equirect_depth_sensor": False,
    "equirect_semantic_sensor": False,
    "seed": 1,
    "silent": False,  # do not print log info (default: OFF)
    # settings exclusive to example.py
    "save_png": False,  # save the pngs to disk (default: OFF)
    "print_semantic_scene": False,
    "print_semantic_mask_stats": False,
    "compute_shortest_path": False,
    "compute_action_shortest_path": False,
    "scene": "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    "test_scene_data_url": "http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip",
    "goal_position": [5.047, 0.199, 11.145],
    "enable_physics": False,
    "enable_gfx_replay_save": False,
    "physics_config_file": "./data/default.physics_config.json",
    "num_objects": 10,
    "test_object_index": 0,
    "frustum_culling": True,
}

# build SimulatorConfiguration`

def make_settings(args, scene):
    settings = default_sim_settings.copy()
    if hasattr(args, "max_frames"):
        settings["max_frames"] = args.max_frames
    if hasattr(args, "width"):
        settings["width"] = args.width
    if hasattr(args, "height"):
        settings["height"] = args.height
    if hasattr(args, "hfov"):
        settings["hfov"] = args.hfov

    # if hasattr(args, "use_vo") and args.use_vo:
    settings["front_width"] = args.front_width
    settings["front_height"] = args.front_height
    settings['front_hfov'] = args.front_hfov


    if hasattr(args, "noisy_rgb"):
        settings["noisy_rgb"] = args.noisy_rgb
        settings["noisy_rgb_multiplier"] = args.noisy_rgb_multiplier
    if hasattr(args, "noisy_depth"):
        settings["noisy_depth"] = args.noisy_depth
        settings["noisy_depth_multiplier"] = args.noisy_depth_multiplier
    if hasattr(args, "noisy_action"):
        settings["noisy_action"] = args.noisy_action

    settings["scene"] = scene
    # settings["save_png"] = args.save_png
    settings["sensor_height"] = args.sensor_height
    settings["color_sensor"] = True
    settings["depth_sensor"] = True
    if hasattr(args, "semantic_sensor") and args.semantic_sensor:
        settings['semantic_sensor'] = True
    settings['move_forward'] = args.move_forward
    settings['act_rot'] = args.act_rot

    if hasattr(args, "add_panoramic_sensor"):
        settings['pano_turn_angle'] = args.panoramic_turn_angle
        settings['add_panoramic_sensor'] = args.add_panoramic_sensor

    # settings["compute_shortest_path"] = args.compute_shortest_path
    # settings["compute_action_shortest_path"] = args.compute_action_shortest_path
    settings["seed"] = args.seed
    settings["silent"] = True
    # settings["enable_physics"] = args.enable_physics
    # settings["physics_config_file"] = args.physics_config_file
    # settings["frustum_culling"] = not args.disable_frustum_culling
    # settings["recompute_navmesh"] = args.recompute_navmesh

    return settings

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    if "scene_dataset_config_file" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    if "frustum_culling" in settings:
        sim_cfg.frustum_culling = settings["frustum_culling"]
    else:
        sim_cfg.frustum_culling = False
    if "enable_physics" in settings:
        sim_cfg.enable_physics = settings["enable_physics"]
    if "physics_config_file" in settings:
        sim_cfg.physics_config_file = settings["physics_config_file"]
    if not settings["silent"]:
        print("sim_cfg.physics_config_file = " + sim_cfg.physics_config_file)
    if "scene_light_setup" in settings:
        sim_cfg.scene_light_setup = settings["scene_light_setup"]
    sim_cfg.gpu_device_id = 0
    if not hasattr(sim_cfg, "scene_id"):
        raise RuntimeError(
            "Error: Please upgrade habitat-sim. SimulatorConfig API version mismatch"
        )
    sim_cfg.scene_id = settings["scene"]


    # define default sensor parameters (see src/esp/Sensor/Sensor.h)
    sensor_specs = []

    def create_camera_spec(**kw_args):
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.resolution = [settings["front_height"], settings["front_width"]]
        camera_sensor_spec.position = [0, settings["sensor_height"], 0]
        for k in kw_args:
            setattr(camera_sensor_spec, k, kw_args[k])
        return camera_sensor_spec

    if settings["color_sensor"]:
        color_sensor_spec = create_camera_spec(
            uuid="color_sensor",
            hfov=settings["front_hfov"],
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        if "noisy_rgb" in settings and settings["noisy_rgb"]:
            color_sensor_spec.noise_model = "GaussianNoiseModel"
            color_sensor_spec.noise_model_kwargs = dict(intensity_constant=settings["noisy_rgb_multiplier"])
        sensor_specs.append(color_sensor_spec)

    if settings["depth_sensor"]:
        depth_sensor_spec = create_camera_spec(
            uuid="depth_sensor",
            hfov=settings["front_hfov"],
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        if "noisy_depth" in settings and settings["noisy_depth"]:
            depth_sensor_spec.noise_model = "RedwoodDepthNoiseModel"
            depth_sensor_spec.noise_model_kwargs = dict(noise_multiplier=settings["noisy_depth_multiplier"])
        sensor_specs.append(depth_sensor_spec)

    if settings["semantic_sensor"]:
        semantic_sensor_spec = create_camera_spec(
            uuid="semantic_sensor",
            hfov=settings["front_hfov"],
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(semantic_sensor_spec)

    if settings["ortho_rgba_sensor"]:
        ortho_rgba_sensor_spec = create_camera_spec(
            uuid="ortho_rgba_sensor",
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
        )
        sensor_specs.append(ortho_rgba_sensor_spec)

    if settings["ortho_depth_sensor"]:
        ortho_depth_sensor_spec = create_camera_spec(
            uuid="ortho_depth_sensor",
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
        )
        sensor_specs.append(ortho_depth_sensor_spec)

    if settings["ortho_semantic_sensor"]:
        ortho_semantic_sensor_spec = create_camera_spec(
            uuid="ortho_semantic_sensor",
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
        )
        sensor_specs.append(ortho_semantic_sensor_spec)

    # TODO Figure out how to implement copying of specs
    def create_fisheye_spec(**kw_args):
        fisheye_sensor_spec = habitat_sim.FisheyeSensorDoubleSphereSpec()
        fisheye_sensor_spec.uuid = "fisheye_sensor"
        fisheye_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        fisheye_sensor_spec.sensor_model_type = (
            habitat_sim.FisheyeSensorModelType.DOUBLE_SPHERE
        )

        # The default value (alpha, xi) is set to match the lens "GoPro" found in Table 3 of this paper:
        # Vladyslav Usenko, Nikolaus Demmel and Daniel Cremers: The Double Sphere
        # Camera Model, The International Conference on 3D Vision (3DV), 2018
        # You can find the intrinsic parameters for the other lenses in the same table as well.
        fisheye_sensor_spec.xi = -0.27
        fisheye_sensor_spec.alpha = 0.57
        fisheye_sensor_spec.focal_length = [364.84, 364.86]

        fisheye_sensor_spec.resolution = [settings["height"], settings["width"]]
        # The default principal_point_offset is the middle of the image
        fisheye_sensor_spec.principal_point_offset = None
        # default: fisheye_sensor_spec.principal_point_offset = [i/2 for i in fisheye_sensor_spec.resolution]
        fisheye_sensor_spec.position = [0, settings["sensor_height"], 0]
        for k in kw_args:
            setattr(fisheye_sensor_spec, k, kw_args[k])
        return fisheye_sensor_spec

    if settings["fisheye_rgba_sensor"]:
        fisheye_rgba_sensor_spec = create_fisheye_spec(uuid="fisheye_rgba_sensor")
        sensor_specs.append(fisheye_rgba_sensor_spec)
    if settings["fisheye_depth_sensor"]:
        fisheye_depth_sensor_spec = create_fisheye_spec(
            uuid="fisheye_depth_sensor",
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
        )
        sensor_specs.append(fisheye_depth_sensor_spec)
    if settings["fisheye_semantic_sensor"]:
        fisheye_semantic_sensor_spec = create_fisheye_spec(
            uuid="fisheye_semantic_sensor",
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
        )
        sensor_specs.append(fisheye_semantic_sensor_spec)

    def create_equirect_spec(**kw_args):
        equirect_sensor_spec = habitat_sim.EquirectangularSensorSpec()
        equirect_sensor_spec.uuid = "equirect_rgba_sensor"
        equirect_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        equirect_sensor_spec.resolution = [settings["equirect_height"], settings["equirect_width"]]
        equirect_sensor_spec.position = [0, settings["sensor_height"], 0]
        for k in kw_args:
            setattr(equirect_sensor_spec, k, kw_args[k])
        return equirect_sensor_spec

    if settings["equirect_rgba_sensor"]:
        equirect_rgba_sensor_spec = create_equirect_spec(uuid="equirect_rgba_sensor")
        sensor_specs.append(equirect_rgba_sensor_spec)

    if settings["equirect_depth_sensor"]:
        equirect_depth_sensor_spec = create_equirect_spec(
            uuid="equirect_depth_sensor",
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
        )
        sensor_specs.append(equirect_depth_sensor_spec)

    if settings["equirect_semantic_sensor"]:
        equirect_semantic_sensor_spec = create_equirect_spec(
            uuid="equirect_semantic_sensor",
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
        )
        sensor_specs.append(equirect_semantic_sensor_spec)


    if "add_panoramic_sensor" in settings and settings["add_panoramic_sensor"]:
        angles = [int(i*settings["pano_turn_angle"]) for i in range(int(360/settings["pano_turn_angle"]))]
        pano_sensors = {}
        for i, r in enumerate(angles):
            pano_sensors[f"rgb_{r}"] = {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
                "orientation": [0, - r * np.pi / 180, 0],
            }

            pano_sensors[f"depth_{r}"] = {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
                "orientation": [0, - r * np.pi / 180, 0],
            }

            pano_sensors[f"semantic_{r}"] = {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
                "orientation": [0, - r * np.pi / 180, 0],
            }


        # create sensor specifications
        for sensor_uuid, sensor_params in pano_sensors.items():

            if 'rgb' in sensor_uuid:
                sensor_spec = create_camera_spec(
                    uuid=sensor_uuid,
                    hfov=settings["hfov"],
                    resolution=sensor_params["resolution"],
                    sensor_type=sensor_params["sensor_type"],
                    sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
                    orientation=sensor_params["orientation"],
                )
                if settings["noisy_rgb"]:
                    sensor_spec.noise_model = "GaussianNoiseModel"
                    sensor_spec.noise_model_kwargs = dict(intensity_constant=settings["noisy_rgb_multiplier"])

            else:
                sensor_spec = create_camera_spec(
                    uuid=sensor_uuid,
                    hfov=settings["hfov"],
                    resolution=sensor_params["resolution"],
                    sensor_type=sensor_params["sensor_type"],
                    channels=1,
                    sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
                    orientation=sensor_params["orientation"],
                )
                if 'depth' in sensor_uuid and settings["noisy_depth"]:
                    sensor_spec.noise_model = "RedwoodDepthNoiseModel"
                    sensor_spec.noise_model_kwargs = dict(noise_multiplier=settings["noisy_depth_multiplier"])

            sensor_specs.append(sensor_spec)




    # create agent specifications
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    if not settings['noisy_action']:
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=settings['move_forward'])
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=settings['act_rot'])
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=settings['act_rot'])
            ),
        }
    else:
        ## -- noisy action -- #

        current_dir = os.path.dirname(os.path.abspath(__file__))
        noise_dir = current_dir + "/noise_models/"
        actuation_noise_fwd = pickle.load(open(noise_dir + "actuation_noise_fwd.pkl", "rb"))
        actuation_noise_right = pickle.load(open(noise_dir + "actuation_noise_right.pkl", "rb"))
        actuation_noise_left = pickle.load(open(noise_dir + "actuation_noise_left.pkl", "rb"))

        def _custom_action_impl(
                scene_node: habitat_sim.SceneNode,
                delta_dist: float,  # in metres
                delta_dist_angle: float,  # in degrees
                delta_angle: float,  # in degrees
        ):
            forward_ax = (
                    np.array(scene_node.absolute_transformation().rotation_scaling())
                    @ habitat_sim.geo.FRONT
            )
            move_angle = np.deg2rad(delta_dist_angle)
            rotation = habitat_sim.utils.quat_from_angle_axis(move_angle, habitat_sim.geo.UP)
            move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)
            scene_node.translate_local(move_ax * delta_dist)
            scene_node.rotate_local(mn.Deg(delta_angle), habitat_sim.geo.UP)

        def _noisy_action_impl(scene_node: habitat_sim.SceneNode, action: int):
            if action == 1:  ## Forward
                dx, dy, do = actuation_noise_fwd.sample()[0][0]
            elif action == 2:  ## Left
                dx, dy, do = actuation_noise_left.sample()[0][0]
            elif action == 3:  ## Right
                dx, dy, do = actuation_noise_right.sample()[0][0]

            delta_dist = np.sqrt(dx ** 2 + dy ** 2)
            delta_dist_angle = np.rad2deg(np.arctan2(-dy, dx))
            delta_angle = -do

            delta_dist = delta_dist * settings['move_forward'] / 0.25         ## noise model assumes 0.25m forward
            delta_angle = delta_angle * settings['act_rot'] / 10    ## noise model assumes 10 degree rotation

            _custom_action_impl(scene_node, delta_dist, delta_dist_angle, delta_angle)

        @habitat_sim.registry.register_move_fn(body_action=True)
        class NoisyForward(habitat_sim.SceneNodeControl):
            def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: int):
                _noisy_action_impl(scene_node, 1)

        @habitat_sim.registry.register_move_fn(body_action=True)
        class NoisyLeft(habitat_sim.SceneNodeControl):
            def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: int):
                _noisy_action_impl(scene_node, 2)

        @habitat_sim.registry.register_move_fn(body_action=True)
        class NoisyRight(habitat_sim.SceneNodeControl):
            def __call__(self, scene_node: habitat_sim.SceneNode, actuation_spec: int):
                _noisy_action_impl(scene_node, 3)


        habitat_sim.registry.register_move_fn(NoisyForward, name="move_forward", body_action=True)
        habitat_sim.registry.register_move_fn(NoisyLeft, name="turn_left", body_action=True)
        habitat_sim.registry.register_move_fn(NoisyRight, name="turn_right", body_action=True)

        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=settings['move_forward'])
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=settings['act_rot'])
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=settings['act_rot'])
            ),
        }

    # override action space to no-op to test physics
    if sim_cfg.enable_physics:
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
            )
        }

    def add_panoramic_turn(turn_panoramic_angle):
        @habitat_sim.registry.register_move_fn(body_action=True)
        class Turn_Panoramic(habitat_sim.SceneNodeControl):
            def __call__(
                    self, scene_node: habitat_sim.SceneNode, actuation_spec: turn_panoramic_angle
            ):
                # Rotate about the +y (up) axis
                rotation_ax = habitat_sim.geo.UP
                scene_node.rotate_local(mn.Deg(turn_panoramic_angle), rotation_ax)
                # Calling normalize is needed after rotating to deal with machine precision errors
                scene_node.rotation = scene_node.rotation.normalized()

        habitat_sim.registry.register_move_fn(
            Turn_Panoramic, name="turn_panoramic", body_action=True
        )

        agent_config = habitat_sim.AgentConfiguration()

        agent_config.action_space["turn_panoramic"] = habitat_sim.ActionSpec(
            "turn_panoramic", turn_panoramic_angle
        )



    if "use_panoramic_turn" in settings:
        add_panoramic_turn(settings["use_panoramic_turn"])




    return habitat_sim.Configuration(sim_cfg, [agent_cfg])




