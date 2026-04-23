"""plotsim — generate realistic multi-table datasets from behavioral archetypes.

Quick start:
    import numpy as np
    from plotsim import load_config, generate_tables, validate, write_tables

    config = load_config("config.yaml")
    tables = generate_tables(config, np.random.default_rng(config.seed))
    report = validate(config, tables)
    write_tables(tables, config, report)
"""

__version__ = "0.3.0"

from plotsim.config import (
    DIRTY,
    NOISE_PRESETS,
    PERFECTLY_CLEAN,
    REALISTIC,
    SLIGHTLY_MESSY,
    PlotsimConfig,
    SurrogateKeyWarning,
    dump_config,
    load_config,
)
from plotsim.output import (
    write_config_copy,
    write_single_table,
    write_tables,
    write_validation_report,
)
from plotsim.tables import generate_tables
from plotsim.validation import (
    ValidationIssue,
    ValidationReport,
    validate_tables as validate,
)
from plotsim.validation import validate_tables

__all__ = [
    "__version__",
    # Config
    "PlotsimConfig",
    "SurrogateKeyWarning",
    "load_config",
    "dump_config",
    "NOISE_PRESETS",
    "PERFECTLY_CLEAN",
    "SLIGHTLY_MESSY",
    "REALISTIC",
    "DIRTY",
    # Generation
    "generate_tables",
    # Validation
    "validate",
    "validate_tables",
    "ValidationReport",
    "ValidationIssue",
    # Output
    "write_tables",
    "write_single_table",
    "write_config_copy",
    "write_validation_report",
]
