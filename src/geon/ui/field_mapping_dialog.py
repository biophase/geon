# # field_mapping_dialog.py

# from __future__ import annotations

# from typing import Dict, List, Tuple

# import numpy as np
# from numpy.typing import NDArray

# from Qt import QtWidgets  # NodeGraphQt uses Qt.py; this resolves to PySide/PyQt
# from NodeGraphQt import NodeGraph  # type: ignore

# from geon.data.pointcloud import FieldType
# from .field_mapping_nodes import PlyFieldNode, OutputFieldNode


# FieldsMap = Dict[str, Tuple[str, int]]        # ply_name -> (field_name, component_index)
# FieldTypes = Dict[str, FieldType]             # field_name -> FieldType


# class FieldMappingDialog(QtWidgets.QDialog):
#     """
#     Visual field-mapping dialog using NodeGraphQt.

#     Inputs:
#       - detected_fields: list of (ply_name, dtype)

#     Interaction:
#       - left side: PLY nodes (one per detected field, one output each)
#       - right side: some pre-created OutputField nodes (Color, Intensity, Semantic, Instance)
#       - right-click on empty canvas: "Add Output -> [Color / Scalar / Vector / Intensity / Semantic / Instance]"

#     Output:
#       - get_mappings() -> (fields_map, field_types)
#         fields_map : ply_name -> (output_field_name, component_index)
#         field_types: output_field_name -> FieldType
#     """

#     def __init__(
#         self,
#         parent: QtWidgets.QWidget | None,
#         detected_fields: List[Tuple[str, str]],
#     ) -> None:
#         super().__init__(parent)
#         self.setWindowTitle("Field Mapping (PLY → PointCloudData)")
#         self.resize(900, 600)

#         self._graph = NodeGraph()
#         self._graph.set_acyclic(True)

#         # register our custom nodes
#         self._graph.register_node(PlyFieldNode)
#         self._graph.register_node(OutputFieldNode)

#         # build UI
#         layout = QtWidgets.QVBoxLayout(self)
#         layout.setContentsMargins(4, 4, 4, 4)

#         layout.addWidget(self._graph.widget)

#         btns = QtWidgets.QDialogButtonBox(
#             QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
#             parent=self,
#         )
#         btns.accepted.connect(self.accept)
#         btns.rejected.connect(self.reject)
#         layout.addWidget(btns)

#         # populate nodes
#         self._build_ply_nodes(detected_fields)
#         self._build_default_output_nodes()
#         self._setup_context_menu()

#     # ------------------------------------------------------------------
#     # graph construction
#     # ------------------------------------------------------------------

#     def _build_ply_nodes(self, detected_fields: List[Tuple[str, str]]) -> None:
#         """
#         Create one PlyFieldNode per detected PLY field on the left side of the graph.
#         """
#         x = -350
#         y = 0
#         dy = 120

#         for name, dtype_str in detected_fields:
#             node: PlyFieldNode = self._graph.create_node(
#                 'geon.fieldmap.PlyFieldNode',
#                 name=name,
#                 pos=(x, y),
#             )
#             node.set_ply_info(name, dtype_str)
#             y += dy

#     def _build_default_output_nodes(self) -> None:
#         """
#         Create some initial output nodes on the right: Color, Intensity, Semantic, Instance.
#         """
#         x = 150
#         y = -150
#         dy = 150

#         defaults: List[Tuple[str, FieldType]] = [
#             ('Color',     FieldType.COLOR),
#             ('Intensity', FieldType.INTENSITY),
#             ('Semantic',  FieldType.SEMANTIC),
#             ('Instance',  FieldType.INSTANCE),
#         ]

#         for name, ftype in defaults:
#             node: OutputFieldNode = self._graph.create_node(
#                 'geon.fieldmap.OutputFieldNode',
#                 name=name,
#                 pos=(x, y),
#             )
#             node.set_field_type(ftype)
#             node.set_field_name(name)
#             y += dy

#     def _setup_context_menu(self) -> None:
#         """
#         Extend the graph context menu with "Add Output → <FieldType>" commands.
#         """
#         graph_menu = self._graph.get_context_menu('graph')

#         add_output_menu = graph_menu.add_menu('Add Output')

#         def make_cmd(field_type: FieldType):
#             # context menu callbacks receive the graph as the only argument.
#             def _cmd(graph: NodeGraph) -> None:
#                 self._add_output_node(field_type)
#             return _cmd

#         # Which types you want the user to be able to add manually.
#         for ftype in [
#             FieldType.COLOR,
#             FieldType.SCALAR,
#             FieldType.VECTOR,
#             FieldType.INTENSITY,
#             FieldType.SEMANTIC,
#             FieldType.INSTANCE,
#         ]:
#             add_output_menu.add_command(
#                 FieldType.get_human_name(ftype),
#                 make_cmd(ftype)
#             )

#     def _add_output_node(
#         self,
#         field_type: FieldType,
#         name: str | None = None,
#     ) -> OutputFieldNode:
#         """
#         Create an OutputFieldNode at the cursor position with a given FieldType.
#         """
#         if name is None:
#             # default name based on type, can be edited later by user
#             name = FieldType.get_human_name(field_type)

#         cx, cy = self._graph.cursor_pos()
#         node: OutputFieldNode = self._graph.create_node(
#             'geon.fieldmap.OutputFieldNode',
#             name=name,
#             pos=(cx, cy),
#         )
#         node.set_field_type(field_type)
#         node.set_field_name(name)
#         return node

#     # ------------------------------------------------------------------
#     # result collection
#     # ------------------------------------------------------------------

#     def get_mappings(self) -> tuple[FieldsMap, FieldTypes]:
#         """
#         Inspect the graph and build:

#             fields_map: ply_name -> (field_name, component_index)
#             field_types: field_name -> FieldType

#         If an output node input has multiple connections, only the first is used.
#         (NodeGraphQt usually enforces 1 connection per input anyway.)
#         """
#         fields_map: FieldsMap = {}
#         field_types: FieldTypes = {}

#         # find all OutputFieldNodes in the graph
#         output_nodes = [
#             n for n in self._graph.all_nodes()
#             if isinstance(n, OutputFieldNode)
#         ]

#         for out_node in output_nodes:
#             field_name = out_node.field_name.strip()
#             if not field_name:
#                 continue  # ignore unnamed outputs

#             ftype = out_node.field_type
#             field_types[field_name] = ftype

#             # inputs will define which PLY fields map into which channel
#             for idx, in_port in enumerate(out_node.inputs()):
#                 # for COLOR/VECTOR we care about component index
#                 # for scalar-like fields it's always index 0
#                 component_index = idx

#                 connected_ports = in_port.connected_ports()
#                 if not connected_ports:
#                     continue

#                 # NodeGraphQt Port.connected_ports() returns a list of ports,
#                 # in our case it's typically [ply_output_port]
#                 src_port = connected_ports[0]
#                 src_node = src_port.node()

#                 if not isinstance(src_node, PlyFieldNode):
#                     # we only care about direct connections from PLY nodes
#                     continue

#                 ply_name = src_node.ply_name
#                 fields_map[ply_name] = (field_name, component_index)

#         return fields_map, field_types
