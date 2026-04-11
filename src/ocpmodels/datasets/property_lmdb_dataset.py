"""Dataset aliases for generic materials property LMDBs."""

from ocpmodels.common.registry import registry

from .lmdb_dataset import LmdbDataset


@registry.register_dataset("property_lmdb")
@registry.register_dataset("materials_property_lmdb")
class PropertyLmdbDataset(LmdbDataset):
    """Thin alias around :class:`LmdbDataset` for generic property tasks."""

    pass
