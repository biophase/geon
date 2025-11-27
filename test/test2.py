from geon.data.pointcloud import PointCloudData, FieldType
from geon.data.document import Document
import numpy as np

doc = Document()



# pcd 1
points = np.ones((43,3),float)
doc.add_data(PointCloudData(points)) 

# pcd 2
points = np.ones((57,3),float)
pcd2 = PointCloudData(points)

doc.add_data(pcd2)
pcd2.add_field('semantics', field_type=FieldType.SEMANTIC)
pcd2.add_field('instance', field_type=FieldType.INSTANCE)


print (doc)

print(doc.items['PCD_0002']['semantics'])
print(doc.items['PCD_0002']['instance'])




