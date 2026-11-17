"""
Concrete :class:`~defog_adapt.core.domain.GraphDomain` implementations for specific
kinds of graphs. These may pull in domain-specific dependencies (e.g. RDKit for
molecules) and are kept out of ``defog_adapt.core`` so the core stays lightweight.
"""
try:
    from .molecule import (
        MoleculeDomain,
        pyg_data_to_mol,
        mol_to_smiles,
        molecular_metrics,
        largest_fragment_smiles,
        validity_report,
        ring_sizes_ok,
        morgan_condition,
        descriptor_values,
        property_distributions,
        continuous_kl,
    )
except ModuleNotFoundError as exc:                 # RDKit is an optional dependency
    if (exc.name or "").split(".")[0] == "rdkit":
        raise ImportError(
            "defog_adapt.domains needs RDKit, which is an optional dependency. Install it with "
            "'pip install \"defog-adapt[chem]\"' (or 'pip install rdkit')."
        ) from exc
    raise

__all__ = [
    "MoleculeDomain",
    "pyg_data_to_mol",
    "mol_to_smiles",
    "molecular_metrics",
    "largest_fragment_smiles",
    "validity_report",
    "ring_sizes_ok",
    "morgan_condition",
    "descriptor_values",
    "property_distributions",
    "continuous_kl",
]
