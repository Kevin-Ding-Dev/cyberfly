# Contributing

See [README.md](README.md) for installation. Run commands from the repository
root with the virtual environment activated.

Before submitting changes, run the checks relevant to the affected components:

```bash
python -m pip check
python -m unittest discover -s tests -v
python -m cyberfly check
python -m cyberfly forage-check
python -m cyberfly maze-check
```

For rendering changes, also record a short video and inspect it locally.
For training changes, run the short pipeline described in
[FORAGING.md](FORAGING.md#3-先确认循环训练能够跑通). A successful update or
completed rollout is not evidence of behavioral improvement.

Include the motivation, affected behavior, verification commands, and remaining
limitations in your pull request. Changes to sensors, rewards, food geometry,
units, or termination conditions must update the configuration and documentation.
Preserve checkpoint compatibility or make incompatible changes explicit.
For maze training changes, also run the automatic-device and explicit MPS smoke tests in [MAZE.md](MAZE.md),
including CPU/MPS checkpoint migration and resume. Calibration must not mutate the
source policy or its optimizer. A device label alone is insufficient: check
parameter, gradient, and optimizer-moment devices. Keep CPU physics distinct
from GPU policy training when reporting performance.

Use millimeters and seconds for simulation quantities unless a field explicitly
states otherwise. Distinguish normalized intake from physical volume or calories.
Support biological claims with primary sources and matched experimental data.

Keep virtual environments, caches, local configuration, raw datasets, checkpoints,
logs, and videos out of source commits. Before publishing output separately,
inspect both its contents and metadata for machine-specific paths or identifiers.
Use a platform-provided private commit email or a project identity when needed;
commit author and committer metadata are published with Git history.

By submitting a contribution, you agree to license it under this project's
Apache-2.0 license. Preserve applicable third-party attribution.
