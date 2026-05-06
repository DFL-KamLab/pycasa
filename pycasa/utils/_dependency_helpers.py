import importlib
import subprocess
import sys
from types import ModuleType


def _is_interactive_terminal() -> bool:
    """Return ``True`` when stdin/stdout are attached to an interactive TTY."""
    stdin = getattr(sys, "stdin", None)
    stdout = getattr(sys, "stdout", None)
    return bool(stdin and stdout and stdin.isatty() and stdout.isatty())


def _ensure_import(
    import_name: str,
    *,
    pip_name: str | None = None,
    prompt_install: bool = True,
    terminate_on_decline: bool = True,
    required: bool = False,
) -> ModuleType:
    """Check/import optional dependency, optionally prompting to install it.

    Parameters:
        import_name (str):
            Import path, for example ``"huggingface_hub"``.
        pip_name (str | None, optional):
            Package name passed to ``pip install``. Defaults to ``import_name``.
        prompt_install (bool, optional):
            If ``True``, ask user in terminal before attempting installation.
        terminate_on_decline (bool, optional):
            If ``True``, raise ``SystemExit`` when user declines installation.

    Returns:
        ModuleType:
            Imported module object.

    Raises:
        ImportError:
            If dependency is missing and could not be installed/imported.
        SystemExit:
            If dependency is missing, user declines install, and
            ``terminate_on_decline=True``.
    """
    try:
        return importlib.import_module(import_name)
    except ImportError as initial_exc:
        package_name = pip_name or import_name
        install_command = f"{sys.executable} -m pip install {package_name}"

        label = "Required" if required else "Optional"
        if not prompt_install:
            raise ImportError(
                f"Missing {label.lower()} dependency '{import_name}'. Install with: {install_command}"
            ) from initial_exc

        if not _is_interactive_terminal():
            raise ImportError(
                f"Missing {label.lower()} dependency '{import_name}' in a non-interactive environment. "
                f"Install with: {install_command}"
            ) from initial_exc

        prompt = (
            f"{label} dependency '{import_name}' is missing. "
            f"Install '{package_name}' now? [y/N]: "
        )
        answer = input(prompt).strip().lower()
        if answer not in {"y", "yes"}:
            message = (
                f"Dependency '{import_name}' is required for this operation. "
                "User declined installation."
            )
            if terminate_on_decline:
                raise SystemExit(message)
            raise ImportError(message) from initial_exc

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ImportError(
                f"Failed to install '{package_name}'. Run manually: {install_command}"
            ) from exc

        importlib.invalidate_caches()
        return importlib.import_module(import_name)
