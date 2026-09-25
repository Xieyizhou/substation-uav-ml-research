"""Bounded heading setpoints; no flight or hardware side effects."""
import math


def slew_heading(previous, target, elapsed, rate=20.0):
    if not all(math.isfinite(v) for v in (previous, target, elapsed, rate)):
        raise ValueError('Non-finite heading input')
    if elapsed < 0 or rate <= 0:
        raise ValueError('Invalid heading timing/rate')
    limit = rate * min(elapsed, .1)
    delta = (target - previous + 180) % 360 - 180
    return (previous + max(-limit, min(limit, delta)) + 180) % 360 - 180
