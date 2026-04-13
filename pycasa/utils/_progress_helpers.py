import sys
from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")

_PROGRESS_ASCII = "-#"
_PROGRESS_COLOUR = "#ff8c00"


def _progress_bar(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str = "item",
    leave: bool = True,
    enabled: bool = True,
) -> Iterable[T]:
    """Wrap an iterable with the shared pycasa progress-bar style.

    Parameters:
        iterable (Iterable[T]):
            The items to iterate over. This is the underlying sequence or
            generator that the progress bar will wrap.
        total (int | None, optional):
            Total number of expected iterations. Pass this when it is known so
            tqdm can render percentages, ETA, and completion accurately.
        desc (str | None, optional):
            Short label shown to the left of the progress bar, such as the
            current pipeline stage name.
        unit (str, optional):
            Singular item label used by tqdm for rate and count display, for
            example ``"frame"`` or ``"item"``.
        leave (bool, optional):
            If ``True``, keep the completed progress bar on screen. If
            ``False``, clear it after completion.
        enabled (bool, optional):
            If ``True``, attempt to show the styled tqdm progress bar. If
            ``False``, return the original iterable unchanged.

    Returns:
        Iterable[T]:
            Either the tqdm-wrapped iterable or the original iterable when
            progress output is disabled or tqdm is unavailable.
    """
    if not enabled:
        return iterable

    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        disable=not sys.stderr.isatty(),
        ascii=_PROGRESS_ASCII,
        colour=_PROGRESS_COLOUR,
        dynamic_ncols=True,
    )
