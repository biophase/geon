"""Dataset map indexing and targeted writes; never reads point arrays."""
from dataclasses import dataclass
from copy import deepcopy
import h5py
from geon.data.definitions import ColorMap
from geon.data.pointcloud import PointCloudData
from geon.version import GEON_FORMAT_VERSION


@dataclass
class MapReference:
    document: object
    point_id: str
    field_name: str
    mapping: object
    field: object = None
    persist: bool = True

    @property
    def group_path(self):
        return f'/document/{self.point_id}/fields/{self.field_name}'


def scan_maps(dataset):
    result = []
    for ref in sorted(dataset.doc_refs, key=lambda r: (r.path or '', r.name)):
        loaded = dataset._loaded_docs.get(ref.name)
        if loaded is not None:
            for point_id, cloud in sorted(loaded.scene_items.items()):
                if isinstance(cloud, PointCloudData):
                    for f in sorted(cloud.get_fields(), key=lambda f: f.name):
                        if f.color_map is not None:
                            result.append(MapReference(ref, point_id, f.name, f.color_map, f))
        elif ref.path:
            with h5py.File(ref.path, 'r') as handle:
                for point_id, group in sorted(handle['document'].items()):
                    if not isinstance(group, h5py.Group) or 'fields' not in group:
                        continue
                    for name, field in sorted(group['fields'].items()):
                        node = field.get('color_map')
                        if node is None:
                            node = field.get('ColorMap')
                        if node is not None:
                            result.append(MapReference(ref, point_id, name, ColorMap.load_h5py(node)))
    return result


def write_reference(ref, mapping, bindings=None, rename_only=False):
    if ref.document.path and ref.persist:
        with h5py.File(ref.document.path, 'r+') as handle:
            group = handle[ref.group_path]
            if rename_only:
                node = group.get('color_map')
                if node is None:
                    node = group['ColorMap']
                node.attrs['name'] = mapping.name
            else:
                mapping.save_h5py(group)
                if bindings is not None:
                    group.attrs['scalar_component'], group.attrs['rgb_components'] = bindings
                handle['document'].attrs['geon_format_version'] = GEON_FORMAT_VERSION
    if ref.field is not None:
        ref.field.color_map = deepcopy(mapping)
        if bindings is not None:
            ref.field.scalar_component, ref.field.rgb_components = bindings
    if ref.document.path is None or not ref.persist:
        from .dataset import RefModState
        ref.document.modState = RefModState.MODIFIED


def resolve_conflicts(dataset):
    refs = scan_maps(dataset)
    used = {ref.mapping.name for ref in refs}
    definitions, assigned, report = {}, {}, []
    for ref in refs:
        name, sig = ref.mapping.name, ref.mapping.signature()
        key = (name, sig)
        if name not in definitions:
            definitions[name] = sig
        if definitions[name] == sig:
            continue
        if key not in assigned:
            suffix = 1
            while f'{name}.{suffix:03}' in used:
                suffix += 1
            assigned[key] = f'{name}.{suffix:03}'
            used.add(assigned[key])
        mapping = deepcopy(ref.mapping)
        mapping.name = assigned[key]
        try:
            write_reference(ref, mapping, rename_only=True)
            report.append(f'{ref.document.name}/{ref.point_id}/{ref.field_name}: {name} → {mapping.name}')
        except (OSError, KeyError) as exc:
            report.append(f'Failed to rename {ref.document.name}/{ref.field_name}: {exc}')
    return report


def unique_maps(dataset):
    maps = {}
    for ref in scan_maps(dataset):
        maps.setdefault((ref.mapping.name, ref.mapping.signature()), deepcopy(ref.mapping))
    return list(maps.values())


def save_map(dataset, original, draft, active_cloud, active_field):
    targets = []
    if original is not None:
        targets = [r for r in scan_maps(dataset)
                   if r.mapping.name == original.name and r.mapping.signature() == original.signature()]
    for docref in dataset.doc_refs:
        doc = dataset._loaded_docs.get(docref.name)
        if doc is not None and doc.scene_items.get(active_cloud.id) is active_cloud:
            if not any(r.document is docref and r.point_id == active_cloud.id and r.field_name == active_field.name for r in targets):
                targets.append(MapReference(docref, active_cloud.id, active_field.name, original, active_field))
    # Check every existing target before making changes. No whole-document saves.
    for target in targets:
        if target.document.path:
            with h5py.File(target.document.path, 'r+') as handle:
                if target.group_path not in handle:
                    if target.field is None:
                        raise ValueError(f'Field no longer exists: {target.group_path}')
                    target.persist = False
    completed = []
    for target in targets:
        try:
            bindings = ((active_field.scalar_component, active_field.rgb_components)
                        if target.field is active_field else None)
            write_reference(target, draft, bindings)
            completed.append(f'{target.document.name}/{target.field_name}')
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f'Saved: {completed}. Failed: {target.document.name}/{target.field_name}: {exc}') from exc
    active_field.color_map = deepcopy(draft)
    return targets
