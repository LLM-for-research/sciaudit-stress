"""SciAudit-Stress core package.

Subpackages are integration points for later issues:
- schemas       : input/prediction validators
- evaluator     : scoring (verdict, evidence, tags, calibration, selective)
- leakage       : forbidden-key scan, metadata/claim probes
- baselines     : B0-B4 reference systems
- construction  : dataset construction utilities (Task 2/3)
"""

__version__ = "0.0.1"
