from geon.data.pointcloud import PointCloudData, SemanticSchema
from geon.data.document import Document
from argparse import ArgumentParser
import colorama
colorama.init()

import numpy as np

from glob import glob
import os
import os.path as osp
from typing import List, Dict, cast, Optional

def _resolve_schema_class_id(schema: SemanticSchema, class_name: str) -> Optional[int]:
    class_name_lower = class_name.strip().lower()
    for sem_cls in schema.semantic_classes:
        if sem_cls.name.lower() == class_name_lower:
            return int(sem_cls.id)
    return None


def analyze_dataset(dataset_fp: str, fix_unknown_to: Optional[str] = None):
    doc_fps =  list(glob(osp.join(dataset_fp,'*.h5')))
    print(f'[Analysis] Found {len(doc_fps)} h5 documents.')
    print(f'[Analysis] Gathering schemas.')
    schemas : Dict[str, SemanticSchema] = {}
    for doc_fp in doc_fps:
        schemas = schemas | SemanticSchema.scan_h5(doc_fp)

    doc_cache: Dict[str, Document] = {}
    dirty_docs: set[str] = set()
        
    # unique schemas
    unq_schemas = {s.signature(): s for s in schemas.values()}
    print(f'[Analysis] Found {len(unq_schemas)} unique schemas: {[s.name for s in unq_schemas.values()]}')



    for uschema in unq_schemas.values():
        mes = f'[Analysis] Analyzing schema "{uschema.name}"'
        print('-'* len(mes))
        print(mes)
        print('-'* len(mes))
        
        for sem_cls in uschema.semantic_classes:
            print(f'[Analysis] {sem_cls}')
    
        # unique annotations
        print(f'[Analysis] Gathering unique annotations')
        unq_classes = []
        occurance_log = {} # cls_id -> list[doc_name]
        schema_ids = {int(sem_cls.id) for sem_cls in uschema.semantic_classes}
        schema_name_by_id = {int(sem_cls.id): sem_cls.name for sem_cls in uschema.semantic_classes}
        schema_mismatch_count = 0
        target_fix_id: Optional[int] = None
        if fix_unknown_to:
            target_fix_id = _resolve_schema_class_id(uschema, fix_unknown_to)
            if target_fix_id is None:
                print(
                    f'\033[31m[FixError] Class "{fix_unknown_to}" not found in schema "{uschema.name}". '
                    'Skipping auto-fix for this schema.\033[0m'
                )
        for bk in schemas.keys():
            # get doc
            doc_name, pc_id, field_name, schema_name = bk.split('/')
            if uschema.name == schema_name:
                if doc_name not in doc_cache:
                    doc_cache[doc_name] = Document.load_hdf5(osp.join(dataset_fp, f"{doc_name}.h5"))
                doc = doc_cache[doc_name]
                print(f'[Analysis] {doc_name}')
                pcd = cast(PointCloudData, doc.scene_items.get(pc_id))
                data = pcd.get_fields(field_name)[0].data
                uids = np.unique(data)

                unknown_ids = sorted(int(uid) for uid in uids if int(uid) not in schema_ids)
                if len(unknown_ids):
                    path_str = f"{doc_name}/{pc_id}/{field_name}"
                    print(
                        f'\033[31m[SchemaError] Field value(s) not present in schema "{uschema.name}" '
                        f'at "{path_str}": {unknown_ids} '
                        f'(run with --fix-unknown <name-to-map-to>)\033[0m'
                    )
                    schema_mismatch_count += 1
                    if target_fix_id is not None:
                        mask = ~np.isin(data, list(schema_ids))
                        unknown_count = int(np.count_nonzero(mask))
                        if unknown_count > 0:
                            data[mask] = int(target_fix_id)
                            dirty_docs.add(doc_name)
                            print(
                                f'\033[33m[Fix] Remapped {unknown_count} value(s) in "{path_str}" '
                                f'to "{fix_unknown_to}" (id={target_fix_id}).\033[0m'
                            )
                            uids = np.unique(data)

                unq_classes.append(uids)
                for uid in uids:
                    occurance_log.setdefault(int(uid), []).append(doc_name)

            
        unq_classes = np.unique(np.concat(unq_classes))
        
        # report result
        for sem_cls in uschema.semantic_classes:
            if sem_cls.id not in unq_classes:
                print(f"\033[31m[Analysis] \"{sem_cls.name}\" has no occurances\033[0m")

        for sem_id, doc_names in occurance_log.items():
            sem_name = schema_name_by_id.get(int(sem_id), f"<unknown:{int(sem_id)}>")
            print(f'[Analysis] "{sem_name}" occurs in {len(doc_names)} documents.')

        if schema_mismatch_count > 0:
            print(f'\033[31m[Analysis] Found {schema_mismatch_count} field(s) with values not present in schema "{uschema.name}".\033[0m')
    
    if dirty_docs:
        for doc_name in sorted(dirty_docs):
            doc_path = osp.join(dataset_fp, f"{doc_name}.h5")
            doc_cache[doc_name].save_hdf5(doc_path)
            print(f'\033[33m[Fix] Saved updated document: {doc_path}\033[0m')
        print(f'\033[33m[Fix] Updated {len(dirty_docs)} document(s).\033[0m')
        
        
        

        
        
    

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="Dataset folder path")
    parser.add_argument(
        "--fix-unknown",
        type=str,
        default=None,
        help='Map unknown semantic IDs to the given class name (for example: "unlabeled").',
    )
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = parse_arguments()
    dataset_fp = args.dataset
    analyze_dataset(dataset_fp, fix_unknown_to=args.fix_unknown)
