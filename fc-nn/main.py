"""
main.py  –  End-to-end pipeline helper.

Typical workflow
----------------
1. Train:
       python main.py train
   or with custom options:
       python main.py train --timesteps 5000000 --n_envs 8

2. Visualise live (3D MuJoCo window):
       python main.py view                            # random actions (test MJCF model)
       python main.py view --model runs/quad_hover/best_model.zip

3. Validate checkpoints in simulation:
       python main.py validate runs/quad_hover/best_model.zip
       python main.py validate runs/quad_hover/best_model.zip --render

4. Export to C header:
       python main.py export runs/quad_hover/best_model.zip

5. Monitor training curves:
       tensorboard --logdir runs/tensorboard

Each sub-command delegates to its own module so you can also call them
directly (see train.py / validate.py / export_weights.py / viewer.py).
"""

import sys


def cmd_train(argv):
    import train as _t
    sys.argv = ["train.py"] + argv
    _t.main()


def cmd_validate(argv):
    import validate as _v
    sys.argv = ["validate.py"] + argv
    _v.main()


def cmd_export(argv):
    import export_weights as _e
    sys.argv = ["export_weights.py"] + argv
    _e.main()


def cmd_view(argv):
    import viewer as _vw
    sys.argv = ["viewer.py"] + argv
    _vw.main()


COMMANDS = {
    "train":    cmd_train,
    "validate": cmd_validate,
    "export":   cmd_export,
    "view":     cmd_view,
}

HELP = """\
usage: python main.py <command> [options]

Commands:
  train       Train the quadrotor policy with PPO  (MuJoCo).
  view        Open interactive 3D MuJoCo viewer.
  validate    Run simulation validation checkpoints.
  export      Export weights to weights.h C header.

Monitoring:
  tensorboard --logdir runs/tensorboard

Run 'python main.py <command> --help' for per-command options.
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(HELP)
        sys.exit(1)

    command = sys.argv[1]
    rest    = sys.argv[2:]
    COMMANDS[command](rest)


if __name__ == "__main__":
    main()
