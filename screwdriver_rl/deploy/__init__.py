"""Deployment utilities for ScrewdriverRL policies.

This package turns a Stage-2 ``deploy.pth`` bundle into a runnable, env-free
controller and maps its 16-D joint targets onto LinkerHand L20/G20 hardware:

  - :mod:`screwdriver_rl.deploy.policy` — ``DeployPolicy``, the proprioception-only
    inference path (actor + proprioceptive-adaptation euler estimator + delta-target
    integration).  Mirrors HORA's ``act_inference`` contract; never consumes
    privileged simulation state.
  - :mod:`screwdriver_rl.deploy.linker_sdk_map` — pure-function mapping from the
    16 policy joints to the LinkerHand SDK's 20 command slots and 0..255 range.
  - :mod:`screwdriver_rl.deploy.deploy_linker` — a thin live ROS/CAN node (lazy
    imports) that ties the two together against real hardware.

Only ``policy`` and ``linker_sdk_map`` are importable without ROS/Isaac.
"""
