"""A report must identify the bytes evaluated, not a later champion."""
from argparse import Namespace
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from cyberfly.forage_learning import compare_forage, load_policy_snapshot
from cyberfly.foraging import ForageConfig


class CheckpointTests(unittest.TestCase):
    def test_report_stays_with_loaded_checkpoint_during_promotion_and_replacement(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            original = b"original checkpoint contents"
            old = root/"old.zip"
            old.write_bytes(original)
            (root/"new.zip").write_bytes(b"new champion")
            (root/"state.json").write_text(json.dumps({"champion":"old.zip"}))
            model = Namespace(cyberfly_forage_config=ForageConfig().to_dict(), num_timesteps=128)
            loaded = []

            def load(stream, **kwargs):
                # Replace the same pathname after it was read but before deserialization.
                old.write_bytes(b"replaced archive")
                loaded.append(stream.read())
                return model

            def evaluate(env, evaluated, seeds, reference):
                self.assertTrue(evaluated is None or evaluated is model)
                (root/"state.json").write_text(json.dumps({"champion":"new.zip"}))
                return dict.fromkeys(("mean_consumed_fraction", "ate_any_rate", "fall_rate",
                                      "escape_rate", "biology"), 0)

            env = Mock()
            args = Namespace(model=str(root), reference=None, episodes=1, seed_start=50,
                             output=str(root/"evaluation.json"))
            with patch("cyberfly.forage_learning.PPO.load", side_effect=load), \
                    patch("cyberfly.forage_learning.ForagingEnv", return_value=env), \
                    patch("cyberfly.forage_learning.evaluate_forage", side_effect=evaluate), \
                    redirect_stdout(StringIO()):
                compare_forage(args)
            report = json.loads((root/"evaluation.json").read_text())
            self.assertEqual(loaded, [original])
            self.assertEqual(report["model"], str(old.resolve()))
            self.assertEqual(report["model_provenance"], {
                "path":str(old.resolve()), "sha256":hashlib.sha256(original).hexdigest(),
                "size_bytes":len(original)})
            env.close.assert_called_once()

    def test_extensionless_checkpoint_and_wrong_model_type(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/"policy.zip"
            path.write_bytes(b"placeholder")
            good = Namespace(cyberfly_forage_config=ForageConfig().to_dict())
            with patch("cyberfly.forage_learning.PPO.load", return_value=good):
                _, _, provenance = load_policy_snapshot(path.with_suffix(""))
            self.assertEqual(provenance["path"], str(path.resolve()))
            with patch("cyberfly.forage_learning.PPO.load", return_value=Namespace()):
                with self.assertRaisesRegex(ValueError, "foraging checkpoint"):
                    load_policy_snapshot(path)


if __name__ == "__main__":
    unittest.main()
