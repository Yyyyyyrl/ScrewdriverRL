"""Deployment utilities for ScrewdriverRL policies.

This package turns a Stage-2 ``deploy.pth`` bundle into a runnable, env-free
controller and maps its 16-D joint targets onto LinkerHand L20/G20 hardware:

  - :mod:`screwdriver_rl.deploy.policy` — ``DeployPolicy``, the proprioception-only
    inference path (actor + proprioceptive-adaptation latent estimator + delta-target
    integration).  Mirrors HORA's ``act_inference`` contract; never consumes
    privileged simulation state.
  - :mod:`screwdriver_rl.deploy.linker_sdk_map` — pure-function mapping from the
    16 policy joints to the LinkerHand SDK's 20 command slots and 0..255 range,
    with a JSON calibration-overlay mechanism for per-hand sign fixes.
  - :mod:`screwdriver_rl.deploy.hw_utils` — stdlib-only helpers (ramp generation,
    state validation, SDK sys.path bootstrap).
  - :mod:`screwdriver_rl.deploy.deploy_linker` — the live node (lazy ROS/SDK
    imports): ramp-to-pregrasp startup, bundle-rate control loop with watchdog,
    hold/release shutdown, CSV recording, offline ``--dry-run``.
  - :mod:`screwdriver_rl.deploy.hand_check` — first-power-on & calibration
    utility (info / echo / ramp / wiggle / pose / roundtrip).

Only ``deploy_linker``'s live transports need ROS or the SDK; everything else
imports with plain torch (``hw_utils``/``linker_sdk_map`` with stdlib alone).
See ``docs/DEPLOY.md`` for the hardware runbook.
"""
