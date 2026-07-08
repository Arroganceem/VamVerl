"""Window + label specs: video end = success, video start = failure (equal counts)."""

from __future__ import annotations

from typing import Any


def success_ends(
    finish: int,
    *,
    window: int,
    pos_near_count: int,
    pos_near_stride: int,
) -> list[int]:
    """Success window ends at the video end: finish, then a few steps before."""
    w = int(window)
    ends = [int(finish)]
    if pos_near_count <= 0:
        return ends
    ps = max(1, int(pos_near_stride))
    for i in range(1, int(pos_near_count) + 1):
        end = int(finish) - i * ps
        if end >= w:
            ends.append(end)
    return ends


def failure_ends_head(
    *,
    window: int,
    count: int,
    stride: int,
    max_end_exclusive: int,
) -> list[int]:
    """Failure window ends at video start; count matches success; no overlap with success zone."""
    w = int(window)
    s = max(1, int(stride))
    limit = int(max_end_exclusive)
    ends: list[int] = []
    end = w
    while len(ends) < int(count) and end < limit:
        ends.append(end)
        end += s
    return ends


# Back-compat aliases used by older call sites / docs.
pos_ends = success_ends


def neg_ends(
    finish: int,
    *,
    window: int,
    stride: int,
    finish_margin_k: int = 0,
    hard_neg_stride: int = 1,
    hard_neg_count: int = 0,
    pos_near_count: int = 0,
    pos_near_stride: int = 1,
) -> list[int]:
    """Equal-count head failures for a would-be success set at finish (compat wrapper)."""
    del finish_margin_k, hard_neg_stride, hard_neg_count
    positive = success_ends(
        finish,
        window=window,
        pos_near_count=pos_near_count,
        pos_near_stride=pos_near_stride,
    )
    if not positive:
        return []
    return failure_ends_head(
        window=window,
        count=len(positive),
        stride=max(1, int(pos_near_stride) if pos_near_stride else stride),
        max_end_exclusive=min(positive),
    )


def _window_specs(
    finish: int,
    complete: bool,
    *,
    window: int,
    stride: int,
    finish_margin_k: int,
    hard_neg_stride: int,
    hard_neg_count: int,
    pos_near_count: int,
    pos_near_stride: int,
) -> list[dict[str, Any]]:
    """Tail success / head failure with the same clip count.

    - label=1: windows at video end (finish and pos_near before finish), only if complete.
    - label=0: same number of windows from video start (no overlap with success zone).

    ``stride`` / hard-neg args are ignored for sampling; failure uses ``pos_near_stride``.
    """
    del finish_margin_k, hard_neg_stride, hard_neg_count, stride
    if finish < window - 1:
        return []

    positive = success_ends(
        finish,
        window=window,
        pos_near_count=pos_near_count,
        pos_near_stride=pos_near_stride,
    )
    if not positive:
        return []

    fail_stride = max(1, int(pos_near_stride))
    # Failure episode: no success labels; still sample equal-count head clips as failure.
    if not complete:
        neg = failure_ends_head(
            window=window,
            count=len(positive),
            stride=fail_stride,
            max_end_exclusive=int(finish) + 1,
        )
        return [{"end": int(end), "label": 0} for end in neg]

    specs: list[dict[str, Any]] = [{"end": int(end), "label": 1} for end in positive]
    neg = failure_ends_head(
        window=window,
        count=len(positive),
        stride=fail_stride,
        max_end_exclusive=min(positive),
    )
    specs.extend({"end": int(end), "label": 0} for end in neg)
    return specs


def train_window_specs(
    finish: int,
    complete: bool,
    *,
    window: int,
    stride: int,
    finish_margin_k: int,
    hard_neg_stride: int,
    hard_neg_count: int,
    pos_near_count: int = 0,
    pos_near_stride: int = 1,
) -> list[dict[str, Any]]:
    return _window_specs(
        finish,
        complete,
        window=window,
        stride=stride,
        finish_margin_k=finish_margin_k,
        hard_neg_stride=hard_neg_stride,
        hard_neg_count=hard_neg_count,
        pos_near_count=pos_near_count,
        pos_near_stride=pos_near_stride,
    )


def val_window_specs(
    finish: int,
    complete: bool,
    *,
    window: int,
    stride: int,
    finish_margin_k: int,
    hard_neg_stride: int,
    hard_neg_count: int,
    pos_near_count: int = 0,
    pos_near_stride: int = 1,
) -> list[dict[str, Any]]:
    return _window_specs(
        finish,
        complete,
        window=window,
        stride=stride,
        finish_margin_k=finish_margin_k,
        hard_neg_stride=hard_neg_stride,
        hard_neg_count=hard_neg_count,
        pos_near_count=pos_near_count,
        pos_near_stride=pos_near_stride,
    )
