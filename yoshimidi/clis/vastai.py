import json
import os
import subprocess
from dataclasses import dataclass

import fire


def run_rsync() -> None:
    instance = _get_instance()
    _run_rsync(instance)


def run_ssh() -> None:
    instance = _get_instance()
    _run_ssh(instance, [])


def run_make(command: str) -> None:
    instance = _get_instance()
    _run_rsync(instance)
    _run_ssh(
        instance,
        [
            "tmux",
            "new-session",
            " && ".join(
                [
                    f"echo Running: {command}",
                    "cd /root/yoshimidi",
                    f"make {command}",
                ]
            ),
        ],
    )


def _run_ssh(instance: "_Instance", commands: list[str]) -> None:
    os.execv(
        "ssh",
        [
            "ssh",
            f"root@{instance.ip}",
            "-p",
            str(instance.port),
            *commands,
        ],
    )


def _run_rsync(instance: "_Instance") -> None:
    subprocess.check_call(
        [
            "rsync",
            "-r",
            "-e",
            f"ssh -p {instance.port}",
            "--filter=:- .gitignore",
            "--filter=- .git",
            ".",
            f"root@{instance.ip}:/root/yoshimidi",
        ]
    )
    subprocess.check_call(
        [
            "rsync",
            "-r",
            "-e",
            f"ssh -p {instance.port}",
            ".env",
            f"root@{instance.ip}:/root/yoshimidi/.env",
        ]
    )


@dataclass
class _Instance:
    ip: str
    port: int


def _get_instance() -> _Instance:
    output = subprocess.check_output(
        ["poetry", "run", "vastai", "show", "instances", "--raw"],
    )
    output_json = json.loads(output.decode())
    assert len(output_json) == 1, f"Expected exactly one instance, found {output_json}"
    return _Instance(
        ip=output_json[0]["ssh_host"],
        port=output_json[0]["ssh_port"],
    )


if __name__ == "__main__":
    fire.Fire(
        dict(
            rsync=run_rsync,
            ssh=run_ssh,
            make=run_make,
        )
    )
