"""IFC4.3 mesh export with portable semantic, object and face-group mappings."""
from functools import lru_cache
from pathlib import Path
import json
import os
import tempfile
import numpy as np

from geon.data.mesh import MeshLevel
from geon.data.pointcloud import FieldType
from geon.util.resources import resource_path

IFC_SCHEMA = "IFC4X3_ADD2"


def _ifc():
    try:
        import ifcopenshell
        return ifcopenshell
    except ImportError as exc:
        raise RuntimeError("IFC export requires IfcOpenShell. Install geon's dependencies with pip install .") from exc


def _inherits(declaration, name):
    while declaration is not None:
        if declaration.name() == name:
            return True
        declaration = declaration.supertype()
    return False


@lru_cache(maxsize=1)
def available_ifc_classes():
    """Concrete geometry-bearing IFC4.3 elements and spatial elements."""
    schema = _ifc().ifcopenshell_wrapper.schema_by_name(IFC_SCHEMA)
    return sorted(d.name() for d in schema.entities() if not d.is_abstract()
                  and (_inherits(d, "IfcElement") or _inherits(d, "IfcSpatialElement")))


def predefined_types(ifc_class):
    declaration = _ifc().ifcopenshell_wrapper.schema_by_name(IFC_SCHEMA).declaration_by_name(ifc_class)
    for attribute in declaration.all_attributes():
        if attribute.name() == "PredefinedType":
            value = attribute.type_of_attribute()
            if hasattr(value, "declared_type"):
                value = value.declared_type()
            if hasattr(value, "enumeration_items"):
                return list(value.enumeration_items())
    return []


def default_mapping():
    return load_mapping(resource_path("ifc_map.json"))


def load_mapping(path):
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or data.get("version") != 1 or data.get("schema") != IFC_SCHEMA:
        raise ValueError("Expected a version 1 IFC4X3_ADD2 mapping.")
    if not isinstance(data.get("classes"), dict):
        raise ValueError("IFC mapping must contain a classes dictionary.")
    return data


def save_mapping(mapping, path):
    Path(path).write_text(json.dumps(mapping, indent=2)+"\n", encoding="utf-8")


def resolve_field(mesh, reference, expected=None):
    if not isinstance(reference, dict) or not reference.get("name"):
        raise ValueError("Choose a mesh field.")
    try:
        level = MeshLevel(reference["level"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Invalid mesh field level in IFC mapping.") from exc
    fields = mesh.get_fields(names=reference["name"], level=level)
    if len(fields) != 1 or expected is not None and fields[0].field_type != expected:
        raise ValueError(f"{level.value}/{reference['name']} is not a {expected.name.lower() if expected else 'valid'} field.")
    return fields[0]


def _semantic_values(mesh, field, faces):
    level = mesh.field_level(field)
    if level == MeshLevel.FACE:
        indices = faces
    elif level == MeshLevel.OBJECT:
        indices = np.searchsorted(mesh.object_ids, np.unique(mesh.face_object_ids[faces]))
    elif level == MeshLevel.VERTEX:
        indices = np.unique(np.concatenate([mesh.faces[i] for i in faces]))
    else:
        lookup = {tuple(edge): i for i, edge in enumerate(mesh.edges)}
        indices = np.unique([lookup[tuple(sorted((int(a), int(b))))] for i in faces
                             for a, b in zip(mesh.faces[i], np.roll(mesh.faces[i], -1))])
    return field.data.reshape(-1)[indices]


def majority(values):
    ids, counts = np.unique(values, return_counts=True)
    if not len(ids):
        raise ValueError("Cannot vote over an empty object.")
    return int(ids[np.argmax(counts)])


def prepare_objects(mesh, mapping):
    if not mesh.faces:
        raise ValueError("The mesh has no faces to export.")
    semantic = resolve_field(mesh, mapping.get("semantic_field"), FieldType.SEMANTIC)
    reference = mapping.get("object_field")
    if reference:
        objects = resolve_field(mesh, reference, FieldType.INSTANCE)
        object_ids = mesh.face_values(objects).reshape(-1)
    elif mesh.field_level(semantic) == MeshLevel.OBJECT:
        object_ids = mesh.face_object_ids
    else:
        raise ValueError("An object separation instance field is required for non-object semantic fields.")
    groups = None
    if mapping.get("face_field"):
        face_field = resolve_field(mesh, mapping["face_field"], FieldType.INSTANCE)
        groups = mesh.face_values(face_field).reshape(-1)
    allowed = set(available_ifc_classes())
    definitions = mapping.get("classes", {})
    output = []
    for object_id in np.unique(object_ids):
        faces = np.flatnonzero(object_ids == object_id)
        semantic_id = majority(_semantic_values(mesh, semantic, faces))
        definition = definitions.get(str(semantic_id))
        if not isinstance(definition, dict) or definition.get("ifc_class") not in allowed:
            raise ValueError(f"Choose a valid IFC4.3 class for semantic ID {semantic_id}.")
        predefined = definition.get("predefined_type")
        if predefined and predefined not in predefined_types(definition["ifc_class"]):
            raise ValueError(f"{predefined} is not a valid predefined type for {definition['ifc_class']}.")
        output.append((int(object_id), semantic_id, faces, definition))
    return semantic, groups, output


def export_ifc(mesh, path, mapping):
    """Write all faces as open tessellations; never claim reconstructed shells are solid."""
    semantic, face_groups, objects = prepare_objects(mesh, mapping)
    ifcopenshell = _ifc()
    import ifcopenshell.api.root as root
    import ifcopenshell.api.unit as unit
    import ifcopenshell.api.context as context
    import ifcopenshell.api.aggregate as aggregate
    import ifcopenshell.api.spatial as spatial
    import ifcopenshell.api.geometry as geometry
    import ifcopenshell.api.pset as pset
    import ifcopenshell.api.style as style
    model = ifcopenshell.file(schema=IFC_SCHEMA)
    model.header.file_name.name = Path(path).name
    project = root.create_entity(model, ifc_class="IfcProject", name=mesh.name)
    metre = unit.add_si_unit(model, unit_type="LENGTHUNIT")
    area = unit.add_si_unit(model, unit_type="AREAUNIT")
    volume = unit.add_si_unit(model, unit_type="VOLUMEUNIT")
    unit.assign_unit(model, units=[metre, area, volume])
    model_context = context.add_context(model, context_type="Model")
    body = context.add_context(model, context_type="Model", context_identifier="Body",
                               target_view="MODEL_VIEW", parent=model_context)
    site = root.create_entity(model, ifc_class="IfcSite", name=mesh.name)
    aggregate.assign_object(model, products=[site], relating_object=project)
    facility = site
    if semantic.schema.name.startswith("BRIDGE"):
        facility = root.create_entity(model, ifc_class="IfcBridge", name=mesh.name)
        aggregate.assign_object(model, products=[facility], relating_object=site)
    created = []
    features = []
    for object_id, semantic_id, indices, definition in objects:
        name = definition.get("name", f"Class {semantic_id}")
        entity = root.create_entity(model, ifc_class=definition["ifc_class"],
                                    predefined_type=definition.get("predefined_type"),
                                    name=f"{name} {object_id}")
        if definition.get("predefined_type") == "USERDEFINED" and hasattr(entity, "ObjectType"):
            entity.ObjectType = str(name)
        center = mesh.vertices[np.unique(np.concatenate([mesh.faces[i] for i in indices]))].mean(axis=0)
        created.append((entity, center))
        if entity.is_a("IfcSurfaceFeature"):
            features.append((entity, center))
        if hasattr(entity, "UsageType") and entity.UsageType is None:
            entity.UsageType = "NOTDEFINED"
        if entity.is_a("IfcSpatialElement"):
            aggregate.assign_object(model, products=[entity], relating_object=facility)
        elif not entity.is_a("IfcSurfaceFeature"):
            spatial.assign_container(model, products=[entity], relating_structure=facility)
        geometry.edit_object_placement(model, product=entity, matrix=np.eye(4))
        partitions = [(None, indices)] if face_groups is None else [
            (int(label), indices[face_groups[indices] == label]) for label in np.unique(face_groups[indices])]
        vertices, faces = [], []
        for _, subset in partitions:
            used = np.unique(np.concatenate([mesh.faces[i] for i in subset]))
            vertices.append(mesh.vertices[used].tolist())
            faces.append([np.searchsorted(used, mesh.faces[i]).tolist() for i in subset])
        # IfcOpenShell converts all vertex lists in a single call to one NumPy
        # array. Face groups can have different vertex counts, so create their
        # items individually and collect them in one object representation.
        representation = None
        items = []
        for group_vertices, group_faces in zip(vertices, faces):
            group_representation = geometry.add_mesh_representation(model, context=body,
                vertices=[group_vertices], faces=[group_faces], unit_scale=1.)
            items.extend(group_representation.Items)
            if representation is None:
                representation = group_representation
            else:
                model.remove(group_representation)
        representation.Items = items
        for item in representation.Items:
            if item.is_a("IfcPolygonalFaceSet"):
                item.Closed = False
        geometry.assign_representation(model, product=entity, representation=representation)
        try:
            color = semantic.schema.by_id(semantic_id).color
        except IndexError:
            color = (204, 204, 204)
        surface_style = style.add_style(model, name=str(name))
        style.add_surface_style(model, style=surface_style, ifc_class="IfcSurfaceStyleShading",
            attributes={"SurfaceColour": dict(zip(("Red", "Green", "Blue"), [float(c)/255 for c in color]))})
        style.assign_representation_styles(model, shape_representation=representation, styles=[surface_style])
        if face_groups is not None:
            for (label, _), item in zip(partitions, representation.Items):
                aspect_rep = model.create_entity("IfcShapeRepresentation", ContextOfItems=body,
                    RepresentationIdentifier="Body", RepresentationType=representation.RepresentationType, Items=[item])
                model.create_entity("IfcShapeAspect", ShapeRepresentations=[aspect_rep],
                    Name=f"{mapping['face_field']['name']}={label}",
                    Description="Face group from geon instance field", ProductDefinitional=True,
                    PartOfProductDefinitionShape=entity.Representation)
        properties = {"ObjectId": object_id, "SemanticId": semantic_id, "SemanticClass": str(name),
                      "SemanticSchema": semantic.schema.name, "SourceMesh": mesh.name,
                      "FaceCount": len(indices)}
        prop = pset.add_pset(model, product=entity, name="GeonClassification")
        pset.edit_pset(model, pset=prop, properties=properties)
    # IFC surface features require an adhesion relation instead of independent
    # spatial containment. Prefer a paving/slab host before other physical parts.
    for feature, center in features:
        physical = [(entity, point) for entity, point in created
                    if entity.is_a("IfcElement") and not entity.is_a("IfcFeatureElement")]
        preferred = [(entity, point) for entity, point in physical
                     if entity.is_a("IfcPavement") or entity.is_a("IfcSlab")]
        candidates = preferred or physical
        if candidates:
            host = min(candidates, key=lambda item: float(np.linalg.norm(item[1]-center)))[0]
        else:
            host = root.create_entity(model, ifc_class="IfcBuildingElementProxy", name="Unspecified surface-feature host")
            spatial.assign_container(model, products=[host], relating_structure=facility)
        model.create_entity("IfcRelAdheresToElement", GlobalId=ifcopenshell.guid.new(),
                            RelatingElement=host, RelatedSurfaceFeatures=[feature])
    from ifcopenshell import validate
    logger = validate.json_logger()
    validate.validate(model, logger)
    if logger.statements:
        message = logger.statements[0]
        raise ValueError(f"The selected IFC classes require additional information: {message.get('message', message)}")
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".ifc", delete=False) as stream:
            temporary = Path(stream.name)
        model.write(str(temporary))
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return model
