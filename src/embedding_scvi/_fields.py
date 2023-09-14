import numpy as np
from anndata import AnnData
from scvi.data.fields._arraylike_field import CategoricalJointField
from scvi.data.fields._dataframe_field import CategoricalDataFrameField


class ExtendableCategoricalDataFrameField(CategoricalDataFrameField):
    EXTENDED_CATEGORIES_KEY = "extended_categories"

    def transfer_field(
        self,
        state_registry: dict,
        adata_target: AnnData,
        **kwargs,
    ) -> dict:
        """Transfer field from registry to target AnnData."""
        new_state_registry = super().transfer_field(
            state_registry,
            adata_target,
            extend_categories=True,
            **kwargs,
        )
        extended_categories = []
        mapping = state_registry[self.CATEGORICAL_MAPPING_KEY].copy()
        for c in np.unique(self._get_original_column(adata_target)):
            if c not in mapping:
                extended_categories.append(c)

        new_state_registry[self.EXTENDED_CATEGORIES_KEY] = extended_categories
        return new_state_registry

    def register_field(self, adata: AnnData) -> dict:
        """Register field."""
        state_registry = super().register_field(adata)
        state_registry[self.EXTENDED_CATEGORIES_KEY] = []
        return state_registry


class ExtendableCategoricalObsField(ExtendableCategoricalDataFrameField):
    """An AnnDataField for categorical .obs attributes in the AnnData data structure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, field_type="obs", **kwargs)


class ExtendableCategoricalJointField(CategoricalJointField):
    EXTENDED_CATEGORIES_DICT_KEY = "extended_categories"

    def transfer_field(
        self,
        state_registry: dict,
        adata_target: AnnData,
        **kwargs,
    ) -> dict:
        """Transfer the field."""
        new_state_registry = super().transfer_field(
            state_registry,
            adata_target,
            extend_categories=True,
            **kwargs,
        )

        source_cat_dict = state_registry[self.MAPPINGS_KEY].copy()
        extended_categories_dict = {}
        for key, mapping in source_cat_dict.items():
            extended_categories = []
            for c in np.unique(getattr(adata_target, self.source_attr_name)[key]):
                if c not in mapping:
                    extended_categories.append(c)
            extended_categories_dict[key] = extended_categories

        new_state_registry[self.EXTENDED_CATEGORIES_DICT_KEY] = extended_categories_dict

        return new_state_registry

    def register_field(self, adata: AnnData) -> dict:
        """Register the field."""
        state_registry = super().register_field(adata)
        state_registry[self.EXTENDED_CATEGORIES_DICT_KEY] = {}
        return state_registry


class ExtendableCategoricalJointObsField(ExtendableCategoricalJointField):
    """An AnnDataField for a collection of categorical .obs fields in the AnnData data structure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, field_type="obsm", **kwargs)
