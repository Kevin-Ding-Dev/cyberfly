"""Invalid JSON settings must fail before they can affect physics or intake."""
from dataclasses import fields
import json
from pathlib import Path
import unittest

import numpy as np

from cyberfly.env import TaskConfig
from cyberfly.foraging import ForageConfig


class ConfigurationTests(unittest.TestCase):
    def test_all_numeric_fields_reject_nonfinite_and_non_numbers(self):
        for factory in (TaskConfig, ForageConfig):
            for field in fields(factory):
                if field.name in ("food_profile", "volatile_food_odor"):
                    continue
                for invalid in (float("nan"), float("inf"), -float("inf"), True, "2", []):
                    with self.subTest(config=factory.__name__, field=field.name, value=invalid):
                        with self.assertRaises(ValueError):
                            factory(**{field.name:invalid})

    def test_integer_boolean_version_and_domain_validation(self):
        for values in ({"food_min":1.5}, {"food_max":2.0}, {"version":1.0},
                       {"food_min":0}, {"food_max":13}, {"food_min":3,"food_max":2},
                       {"volatile_food_odor":1}, {"version":2}, {"food_profile":"unknown"},
                       {"spawn_min":0}, {"spawn_max":25}, {"sensor_noise":-1},
                       {"residual_scale":0.6}, {"residual_scale":-0.1},
                       {"mouth_residual_scale":2.01}, {"mouth_residual_scale":-0.1},
                       {"sip_seconds":0}, {"ingestion_rate":0}, {"visual_range":0},
                       {"food_radius":0}, {"droplet_volume_ul":0},
                       {"episode_seconds":0.03}, {"episode_seconds":0},
                       {"food_profile":"sucrose-droplet","droplet_volume_ul":20}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ForageConfig(**values)
        for values in ({"version":2}, {"version":1.0}, {"physics_dt":0},
                       {"control_dt":0.00001}, {"episode_seconds":0.03},
                       {"speed_min":0}, {"speed_max":1}, {"heading_range":-1},
                       {"action_scale":-1}, {"action_scale":2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TaskConfig(**values)

    def test_presets_and_legacy_configuration_roundtrip(self):
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root/"configs").glob("forage*.json")):
            with self.subTest(preset=path.name):
                config = ForageConfig(**json.loads(path.read_text()))
                self.assertEqual(ForageConfig(**config.to_dict()), config)
        legacy = ForageConfig(residual_scale=0.3).to_dict()
        self.assertNotIn("mouth_residual_scale", legacy)
        self.assertEqual(ForageConfig(**legacy).to_dict(), legacy)
        self.assertEqual(TaskConfig(**TaskConfig().to_dict()), TaskConfig())

    def test_legacy_actions_keep_original_control_ranges(self):
        base = np.array([0.5, 0.8, -1], dtype=np.float32)
        action = np.array([-1, 1, 1], dtype=np.float32)
        for scale, expected in ((0, [0.5, 0.8, -1]), (0.15, [0.35, 0.95, -0.85]),
                                (0.3, [0.2, 1, -0.7]), (0.5, [0, 1, -0.5])):
            with self.subTest(legacy_scale=scale):
                config = ForageConfig(residual_scale=scale)
                np.testing.assert_allclose(config.motor_command(base, action), expected, atol=1e-7)

    def test_optional_mouth_control_can_open_and_close_without_changing_legs(self):
        config = ForageConfig(mouth_residual_scale=2)
        for mouth_base in (-1, 1):
            base = np.array([0.5, 0.8, mouth_base], dtype=np.float32)
            np.testing.assert_array_equal(config.motor_command(base, np.zeros(3)), base)
            for command, expected in ((1, 1), (-1, -1)):
                motor = config.motor_command(base, np.array([0, 0, command]))
                self.assertEqual(motor[2], expected)
                np.testing.assert_array_equal(motor[:2], base[:2])


if __name__ == "__main__":
    unittest.main()
