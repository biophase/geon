from geon.data.pointcloud import PointCloudData, SemanticSchema
from geon.data.document import Document
from argparse import ArgumentParser
import colorama
colorama.init()

import numpy as np

from glob import glob
import os
import os.path as osp
from typing import List, Dict, cast

def analyze_dataset(dataset_fp: str):
    doc_fps =  list(glob(osp.join(dataset_fp,'*.h5')))
    print(f'[Analysis] Found {len(doc_fps)} h5 documents.')
    print(f'[Analysis] Gathering schemas.')
    schemas : Dict[str, SemanticSchema] = {}
    for doc_fp in doc_fps:
        schemas = schemas | SemanticSchema.scan_h5(doc_fp)
        
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
        for bk in schemas.keys():
            # get doc
            doc_name, pc_id, field_name, schema_name = bk.split('/')
            if uschema.name == schema_name:
                doc = Document.load_hdf5(osp.join(dataset_fp, f"{doc_name}.h5"))
                print(f'[Analysis] {doc_name}')
                pcd = cast(PointCloudData, doc.scene_items.get(pc_id))
                data = pcd.get_fields(field_name)[0].data
                uids = np.unique(data)
                unq_classes.append(uids)
                for uid in uids:
                    occurance_log.setdefault(uid, []).append(doc_name)

            
        unq_classes = np.unique(np.concat(unq_classes))
        
        # report result
        for sem_cls in uschema.semantic_classes:
            if sem_cls.id not in unq_classes:
                print(f"\033[31m[Analysis] \"{sem_cls.name}\" has no occurances\033[0m")

        for sem_id, doc_names in occurance_log.items():
            print(f'[Analysis] "{uschema.by_id(sem_id).name }" occurs in {len(doc_names)} documents.')
        
        
        

        
        
    

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="Dataset folder path")
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = parse_arguments()
    dataset_fp = args.dataset
    analyze_dataset(dataset_fp)