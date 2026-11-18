"""Shared field/schema policies for disjoint data merges."""
from copy import deepcopy
import numpy as np
from .pointcloud import FieldType, SemanticSchema


def merge_schemas(schemas):
    """Align classes by name, keeping the first colour and unused original IDs."""
    classes=[];by_name={};used=set();maps=[]
    for schema in schemas:
        mapping={}
        for item in ([] if schema is None else schema.semantic_classes):
            key=item.name if item.id>=0 else '_unassigned'
            if key not in by_name:
                clone=deepcopy(item)
                if clone.id in used:
                    clone.id=max([0,*[i+1 for i in used if i>=0]])
                classes.append(clone);used.add(clone.id);by_name[key]=clone.id
            mapping[item.id]=by_name[key]
        maps.append(mapping)
    first=next((s for s in schemas if s is not None),None)
    return (SemanticSchema(first.name,classes) if first else None),maps


def merge_fields(containers):
    """Union fields by name/type/width; missing labels=-1, colour=0, numeric=NaN.

    Nonnegative instance IDs are compacted into disjoint ranges per input,
    independently for each output field. Negative labels remain unassigned.
    Semantic IDs are aligned by class name. Incompatible fields are retained
    under unique suffixed names.
    """
    groups={};reserved={f.name for c in containers for f in c.get_fields()};used=set()
    for index,c in enumerate(containers):
        for f in c.get_fields():
            data=np.asarray(f.data)
            columns=int(np.prod(data.shape[1:])) if data.ndim>1 else 1
            key=(f.name,f.field_type,columns)
            groups.setdefault(key,{})[index]=f
    result=[]
    for (name,kind,columns),members in groups.items():
        first=next(iter(members.values()));field=deepcopy(first)
        if name in used:
            suffix=1
            while f'{name}_{suffix:03}' in used|reserved:suffix+=1
            name=f'{name}_{suffix:03}'
        used.add(name);field.name=name
        missing=len(members)!=len(containers)
        categorical=kind in (FieldType.INSTANCE,FieldType.SEMANTIC)
        dtype=np.result_type(*[f.data.dtype for f in members.values()])
        if categorical:dtype=np.dtype('int64')
        elif missing and kind!=FieldType.COLOR:dtype=np.result_type(dtype,np.float64)
        maps=[{} for _ in containers]
        if kind==FieldType.SEMANTIC:
            field.schema,maps=merge_schemas([members[i].schema if i in members else None for i in range(len(containers))])
        chunks=[]
        next_instance_id=0
        for i,c in enumerate(containers):
            if i not in members:
                fill=-1 if categorical else (0 if kind==FieldType.COLOR else np.nan)
                chunks.append(np.full((len(c.points),columns),fill,dtype=dtype));continue
            data=np.asarray(members[i].data).reshape(len(c.points),columns).astype(dtype,copy=True)
            if kind==FieldType.INSTANCE:
                assigned=data>=0
                source_ids,inverse=np.unique(data[assigned],return_inverse=True)
                data[assigned]=inverse+next_instance_id
                next_instance_id+=len(source_ids)
            if kind==FieldType.SEMANTIC:
                original=data.copy()
                for old,new in maps[i].items():data[original==old]=new
            chunks.append(data)
        field.data=np.concatenate(chunks)
        result.append(field)
    return result
