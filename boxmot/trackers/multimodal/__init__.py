"""Trackers whose state fundamentally combines multiple representations.

Implementations may consume boxes, masks, prompts, or tracker-specific model
memory. General-purpose perception inference remains owned by the detector,
segmentor, and ReID domains. Import public tracker classes from :mod:`boxmot`.
"""
