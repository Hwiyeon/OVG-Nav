#!/usr/bin/env python3
from typing import Any
from gym import spaces
import habitat
from habitat.core.simulator import SemanticSensor
from habitat.core.simulator import RGBSensor
from habitat.core.simulator import DepthSensor
import numpy as np
import habitat_sim

import matplotlib.pyplot as plt

RGBSENSOR_DIMENSION = 3


def check_sim_obs(obs, sensor):
    assert obs is not None, (
        "Observation corresponding to {} not present in "
        "simulator's observations".format(sensor.uuid)
    )


@habitat.registry.register_sensor(name="angle_rgb_sensor")
class Angle_RGBSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        self.angle = config.ANGLE
        # print(self.angle)
        super().__init__(config=config)
        self.config = config

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "rgb_" + str(self.angle)

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs


@habitat.registry.register_sensor(name="angle_depth_sensor")
class Angle_DepthSensor(DepthSensor):
    sim_sensor_type: habitat_sim.SensorType
    min_depth_value: float
    max_depth_value: float

    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.angle = config.ANGLE
        self.config = config
        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH

        super().__init__(config=config)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "depth_" + str(self.angle)

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.DEPTH

    # This is called whenever reset is called or an action is taken
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32,
        )

    def get_observation(self, sim_obs):
        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)

        obs = np.expand_dims(
            obs, axis=2
        )  # make depth observation a 3D array
        if self.config.NORMALIZE_DEPTH:
            # normalize depth observation to [0, 1]
            obs = (obs - self.config.MIN_DEPTH) / (
                    self.config.MAX_DEPTH - self.config.MIN_DEPTH
            )
        return obs




@habitat.registry.register_sensor(name="panoramic_rgb_sensor")
class Pano_RGBSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        # print(self.angle)
        super().__init__(config=config)
        self.config = config

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "panoramic_rgb"

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, RGBSENSOR_DIMENSION),
            dtype=np.uint8,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, sim_obs):

        config.SIMULATOR.ANGLE_RGB_SENSOR_0 = config.SIMULATOR.RGB_SENSOR.clone()
        config.SIMULATOR.ANGLE_RGB_SENSOR_0.TYPE = "angle_rgb_sensor"
        config.SIMULATOR.ANGLE_RGB_SENSOR_0.ANGLE = 0
        config.SIMULATOR.ANGLE_RGB_SENSOR_0.ORIENTATION = [0, 0, 0]
        config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_RGB_SENSOR_0")

        config.SIMULATOR.ANGLE_RGB_SENSOR_90 = config.SIMULATOR.RGB_SENSOR.clone()
        config.SIMULATOR.ANGLE_RGB_SENSOR_90.TYPE = "angle_rgb_sensor"
        config.SIMULATOR.ANGLE_RGB_SENSOR_90.ANGLE = 90
        config.SIMULATOR.ANGLE_RGB_SENSOR_90.ORIENTATION = [0, -90 * np.pi / 180, 0]
        config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_RGB_SENSOR_90")

        config.SIMULATOR.ANGLE_RGB_SENSOR_180 = config.SIMULATOR.RGB_SENSOR.clone()
        config.SIMULATOR.ANGLE_RGB_SENSOR_180.TYPE = "angle_rgb_sensor"
        config.SIMULATOR.ANGLE_RGB_SENSOR_180.ANGLE = 180
        config.SIMULATOR.ANGLE_RGB_SENSOR_180.ORIENTATION = [0, -180 * np.pi / 180, 0]
        config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_RGB_SENSOR_180")

        config.SIMULATOR.ANGLE_RGB_SENSOR_270 = config.SIMULATOR.RGB_SENSOR.clone()
        config.SIMULATOR.ANGLE_RGB_SENSOR_270.TYPE = "angle_rgb_sensor"
        config.SIMULATOR.ANGLE_RGB_SENSOR_270.ANGLE = 270
        config.SIMULATOR.ANGLE_RGB_SENSOR_270.ORIENTATION = [0, -270 * np.pi / 180, 0]
        config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_RGB_SENSOR_270")

        obs = sim_obs.get(self.uuid, None)
        check_sim_obs(obs, self)

        # remove alpha channel
        obs = obs[:, :, :RGBSENSOR_DIMENSION]
        return obs


def panoramic_90(config, semantic=False):
    """
    Add panoramic RGB, Depth sensor with 90 degrees
    For the orientation, see habitat-api/habitat/sims/habitat_simulator/habitat_simulator.py --- create_sim_config
    """
    config.defrost()

    # Add RGB sensors
    config.SIMULATOR.ANGLE_RGB_SENSOR_0 = config.SIMULATOR.RGB_SENSOR.clone()
    config.SIMULATOR.ANGLE_RGB_SENSOR_0.TYPE = "angle_rgb_sensor"
    config.SIMULATOR.ANGLE_RGB_SENSOR_0.ANGLE = 0
    config.SIMULATOR.ANGLE_RGB_SENSOR_0.ORIENTATION = [0, 0, 0]
    config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_RGB_SENSOR_0")

    config.SIMULATOR.ANGLE_RGB_SENSOR_90 = config.SIMULATOR.RGB_SENSOR.clone()
    config.SIMULATOR.ANGLE_RGB_SENSOR_90.TYPE = "angle_rgb_sensor"
    config.SIMULATOR.ANGLE_RGB_SENSOR_90.ANGLE = 90
    config.SIMULATOR.ANGLE_RGB_SENSOR_90.ORIENTATION = [0, -90 * np.pi / 180, 0]
    config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_RGB_SENSOR_90")

    config.SIMULATOR.ANGLE_RGB_SENSOR_180 = config.SIMULATOR.RGB_SENSOR.clone()
    config.SIMULATOR.ANGLE_RGB_SENSOR_180.TYPE = "angle_rgb_sensor"
    config.SIMULATOR.ANGLE_RGB_SENSOR_180.ANGLE = 180
    config.SIMULATOR.ANGLE_RGB_SENSOR_180.ORIENTATION = [0, -180 * np.pi / 180, 0]
    config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_RGB_SENSOR_180")

    config.SIMULATOR.ANGLE_RGB_SENSOR_270 = config.SIMULATOR.RGB_SENSOR.clone()
    config.SIMULATOR.ANGLE_RGB_SENSOR_270.TYPE = "angle_rgb_sensor"
    config.SIMULATOR.ANGLE_RGB_SENSOR_270.ANGLE = 270
    config.SIMULATOR.ANGLE_RGB_SENSOR_270.ORIENTATION = [0, -270 * np.pi / 180, 0]
    config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_RGB_SENSOR_270")

    ## Add depth sensors

    config.SIMULATOR.ANGLE_DEPTH_SENSOR_90 = config.SIMULATOR.DEPTH_SENSOR.clone()
    config.SIMULATOR.ANGLE_DEPTH_SENSOR_90.TYPE = "angle_depth_sensor"
    config.SIMULATOR.ANGLE_DEPTH_SENSOR_90.ANGLE = 90
    config.SIMULATOR.ANGLE_DEPTH_SENSOR_90.ORIENTATION = [0, -90 * np.pi / 180, 0]
    config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_DEPTH_SENSOR_90")

    config.SIMULATOR.ANGLE_DEPTH_SENSOR_180 = config.SIMULATOR.DEPTH_SENSOR.clone()
    config.SIMULATOR.ANGLE_DEPTH_SENSOR_180.TYPE = "angle_depth_sensor"
    config.SIMULATOR.ANGLE_DEPTH_SENSOR_180.ANGLE = 180
    config.SIMULATOR.ANGLE_DEPTH_SENSOR_180.ORIENTATION = [0, -180 * np.pi / 180, 0]
    config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_DEPTH_SENSOR_180")

    config.SIMULATOR.ANGLE_DEPTH_SENSOR_270 = config.SIMULATOR.DEPTH_SENSOR.clone()
    config.SIMULATOR.ANGLE_DEPTH_SENSOR_270.TYPE = "angle_depth_sensor"
    config.SIMULATOR.ANGLE_DEPTH_SENSOR_270.ANGLE = 270
    config.SIMULATOR.ANGLE_DEPTH_SENSOR_270.ORIENTATION = [0, -270 * np.pi / 180, 0]
    config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_DEPTH_SENSOR_270")

    config.freeze()


if __name__ == "__main__":
    # Get the default config node
    config = habitat.get_config("objectnav_mp3d_c2om.yaml")
    config.defrost()

    config.DATASET.DATA_PATH = '/home/hwing/Dataset/habitat/data/datasets/pointnav_mp3d_v1/val_mini/val_mini.json.gz'
    config.DATASET.SCENES_DIR = '/home/hwing/Dataset/habitat/data/scene_datasets'

    config.SIMULATOR.ANGLE_RGB_SENSOR1 = config.SIMULATOR.RGB_SENSOR.clone()
    config.SIMULATOR.ANGLE_RGB_SENSOR1.TYPE = "angle_rgb_sensor"
    config.SIMULATOR.ANGLE_RGB_SENSOR1.ANGLE = config.SIMULATOR.TURN_ANGLE
    config.SIMULATOR.ANGLE_RGB_SENSOR1.ORIENTATION = [0, 30 * np.pi / 180, 0]
    #### See habitat-api/habitat/sims/habitat_simulator/habitat_simulator.py --- create_sim_config
    config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_RGB_SENSOR1")

    config.SIMULATOR.ANGLE_RGB_SENSOR2 = config.SIMULATOR.RGB_SENSOR.clone()
    config.SIMULATOR.ANGLE_RGB_SENSOR2.TYPE = "angle_rgb_sensor"
    config.SIMULATOR.ANGLE_RGB_SENSOR2.ANGLE = 60
    config.SIMULATOR.AGENT_0.SENSORS.append("ANGLE_RGB_SENSOR2")
    config.freeze()

    env = habitat.Env(config=config)
    print(env.reset()["agent_position"])
    print(env.get_metrics()["episode_info"])
    print(env.step("MOVE_FORWARD")["agent_position"])
    print(env.get_metrics()["episode_info"])
