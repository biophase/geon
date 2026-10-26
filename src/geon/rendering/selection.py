"""Optional capability for layers with selectable subelements.

Selections retain each layer's own representation. Extraction copies data and
returns an unattached, unselected layer; inserting it is a controller operation.
"""
from abc import ABC, abstractmethod
from copy import deepcopy


class SelectableLayer(ABC):
    @property
    @abstractmethod
    def active_selection(self):
        ...

    @active_selection.setter
    @abstractmethod
    def active_selection(self, value):
        ...

    @property
    def has_selection(self) -> bool:
        selection = self.active_selection
        return selection is not None and len(selection) > 0

    def clear_selection(self) -> None:
        """Clear selected elements and all transient selection state."""
        self.active_selection = None

    def capture_selection(self):
        """Independent snapshot for undo; subclasses include auxiliary state."""
        return deepcopy(self.active_selection)

    def restore_selection(self, state) -> None:
        self.active_selection = deepcopy(state)

    def extract_selection(self):
        """Copy selected data, including required topology and all attributes.

        The source is unchanged. Empty selections are rejected. Element IDs
        remain local to the new data object; point row mappings are available
        through PointCloudData.extract_subcloud for algorithm callers.
        """
        if not self.has_selection:
            raise ValueError("The layer has no selected elements.")
        data = self.data.extract_selection(self.active_selection)
        layer = type(self)(data)
        layer.browser_name = f"{self.browser_name} (selection)"
        return layer
