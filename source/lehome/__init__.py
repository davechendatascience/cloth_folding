"""Namespace shim.

In the real deployment this is LeHome's own ``source/lehome`` package. Here it
only needs to make ``lehome.real_damped_project`` importable from a checkout.
When installing into ``lehome-challenge``, drop ``real_damped_project/`` into
the existing ``source/lehome/`` and delete this file.
"""
from __future__ import annotations

__path__ = __import__("pkgutil").extend_path(__path__, __name__)
