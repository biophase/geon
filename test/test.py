import pytest
import numpy as np
import json
import os
from dataclasses import asdict

# ADJUST THIS IMPORT to match your actual file structure
# e.g. from geon.core.models import ...
from geon.data.pointcloud import (
    SemanticSchema, 
    SemanticClass, 
    SemanticSegmentation, 
    InstanceSegmentation,
    PointCloudData,
    mex,
    FieldType
)

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTION TESTS
# -----------------------------------------------------------------------------

def test_mex_logic():
    """
    Test the Minimum Excluded value logic.
    Critical for generating new Instance IDs automatically.
    """
    # Standard sequence
    assert mex(np.array([0, 1, 2])) == 3
    # Gap in sequence
    assert mex(np.array([0, 2, 3])) == 1
    # Unordered
    assert mex(np.array([5, 0, 1])) == 2
    # Empty array
    assert mex(np.array([])) == 0
    # Only negatives (unlabeled) should return 0
    assert mex(np.array([-1, -1, -5])) == 0

# -----------------------------------------------------------------------------
# 2. SEMANTIC SCHEMA TESTS
# -----------------------------------------------------------------------------

@pytest.fixture
def basic_schema():
    """Creates a schema with just the default _unlabeled class"""
    return SemanticSchema()

def test_schema_initialization(basic_schema):
    """Ensure schema starts with the default unlabeled class"""
    assert len(basic_schema.semantic_classes) == 1
    assert basic_schema.semantic_classes[0].id == -1
    assert basic_schema.semantic_classes[0].name == '_unlabeled'

def test_add_semantic_class(basic_schema):
    """Test adding classes and preventing duplicates"""
    new_class = SemanticClass(id=1, name="Wall", color=(1.0, 0.0, 0.0))
    basic_schema.add_semantic_class(new_class)
    
    assert len(basic_schema.semantic_classes) == 2
    assert basic_schema.semantic_classes[1].name == "Wall"

    # Test Duplicate ID Rejection
    duplicate_id = SemanticClass(id=1, name="Floor", color=(0.0, 1.0, 0.0))
    with pytest.raises(AssertionError) as excinfo:
        basic_schema.add_semantic_class(duplicate_id)
    assert "Index 1 already exists" in str(excinfo.value)

    # Test Duplicate Name Rejection
    duplicate_name = SemanticClass(id=2, name="Wall", color=(0.0, 0.0, 1.0))
    with pytest.raises(AssertionError) as excinfo:
        basic_schema.add_semantic_class(duplicate_name)
    assert "Name Wall already exists" in str(excinfo.value)

def test_schema_json_io(basic_schema, tmp_path):
    """Test saving and loading the schema from JSON"""
    # Setup
    c1 = SemanticClass(10, "Column", (0.5, 0.5, 0.5))
    basic_schema.add_semantic_class(c1)
    
    file_path = tmp_path / "schema_test.json"
    
    # Save
    basic_schema.to_json(str(file_path))
    
    # Check file exists
    assert os.path.exists(file_path)
    
    # Load back
    loaded_schema = SemanticSchema.from_json(str(file_path))
    
    # Verify content
    assert len(loaded_schema.semantic_classes) == 2
    # Note: JSON loads tuples as lists, so we compare values
    loaded_class = loaded_schema.semantic_classes[1]
    assert loaded_class.id == 10
    assert loaded_class.name == "Column"
    assert list(loaded_class.color) == [0.5, 0.5, 0.5]

# -----------------------------------------------------------------------------
# 3. SEGMENTATION DATA STRUCTURE TESTS
# -----------------------------------------------------------------------------

def test_semantic_seg_initialization(basic_schema):
    """Test that we can create a segmentation from data OR size"""
    
    # Case A: From Data
    data = np.array([0, 1, 0, -1])
    seg = SemanticSegmentation("test_seg", data=data, size=None, schema=basic_schema)
    assert seg.field_type == FieldType.SEMANTIC
    assert np.array_equal(seg.data, data)
    
    # Case B: From Size
    seg_empty = SemanticSegmentation("empty_seg", data=None, size=100, schema=basic_schema)
    assert len(seg_empty.data) == 100
    assert seg_empty.data[0] == -1 # Should default to -1 (unlabeled)
    
    # Case C: Error (Both None)
    with pytest.raises(ValueError):
        SemanticSegmentation("bad", data=None, size=None, schema=basic_schema)

def test_instance_seg_next_id():
    """Test generating the next available instance ID"""
    data = np.array([1, 1, 2, 5])
    inst_seg = InstanceSegmentation("inst", data=data, size=None)
    
    # mex of 1,2,5 is 0? Wait, instances usually start at 1 or 0?
    # Based on mex logic: 0 is missing from [1, 1, 2, 5]
    # If 0 is background, next valid object ID might be expected to be > 0.
    # But strictly speaking, your mex function returns 0 here.
    assert inst_seg.get_next_instance_id() == 0 
    
    # If we have 0, 1, 2
    data2 = np.array([0, 1, 2])
    inst_seg2 = InstanceSegmentation("inst2", data=data2, size=None)
    assert inst_seg2.get_next_instance_id() == 3

# -----------------------------------------------------------------------------
# 4. COMPLEX LOGIC: REINDEXING (The TDD Part)
# -----------------------------------------------------------------------------

def test_reindexing_sync():
    """
    Test that reindexing the schema ALSO updates the numpy array.
    This ensures data consistency.
    """
    # 1. Setup specific scenario
    # We have IDs 100 and 200. We want them to become 0 and 1.
    schema = SemanticSchema()
    schema.add_semantic_class(SemanticClass(100, "Window", (0,1,1)))
    schema.add_semantic_class(SemanticClass(200, "Door", (1,0,1)))
    
    # Create dummy points: two points are Windows (100), two are Doors (200), one is Unlabeled (-1)
    raw_data = np.array([100, 100, 200, 200, -1]) 
    
    seg = SemanticSegmentation("gt", data=raw_data, size=None, schema=schema)
    
    # 2. Execute Reindex
    # NOTE: This will fail until you implement the method in SemanticSegmentation!
    try:
        seg.remap()
    except NotImplementedError:
        pytest.fail("reindex_semantic is not implemented yet!")
        
    # 3. Verify Schema Changes
    # IDs should be sorted: -1 remains -1. 100 becomes 0. 200 becomes 1.
    assert schema.semantic_classes[0].id == -1
    assert schema.semantic_classes[1].id == 0
    assert schema.semantic_classes[1].name == "Window"
    assert schema.semantic_classes[2].id == 1
    assert schema.semantic_classes[2].name == "Door"
    
    # 4. Verify Data Changes (The most critical part)
    expected_data = np.array([0, 0, 1, 1, -1])
    np.testing.assert_array_equal(seg.data, expected_data, 
                                  err_msg="Numpy array was not updated to match new Schema IDs")

# -----------------------------------------------------------------------------
# 5. INTEGRATION: POINT CLOUD DATA
# -----------------------------------------------------------------------------

def test_point_cloud_container():
    """Test the main container holder"""
    points = np.random.rand(100, 3)
    pc = PointCloudData(points)
    
    assert pc.points.shape == (100, 3)
    assert len(pc.segmentations) == 0
    
    # Simulate adding a segmentation
    schema = SemanticSchema()
    seg = SemanticSegmentation("base", size=100, data=None, schema=schema)
    pc.segmentations.append(seg)
    
    assert len(pc.segmentations) == 1
    assert pc.segmentations[0].name == "base"