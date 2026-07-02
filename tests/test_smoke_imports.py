def test_imports():
    import geon  # noqa: F401
    from geon._native import features  # noqa: F401
    from geon._native import plane_ransac  # noqa: F401
    from geon._native import region_merge  # noqa: F401
    from geon._native import region_growing  # noqa: F401
    from geon._native import superpoints  # noqa: F401
    from geon.data.boundingbox import BoundingBoxData  # noqa: F401
    from geon.rendering.boundingbox import BoundingBoxLayer  # noqa: F401
    from geon.tools.boundingbox import CreateHorizontalBoundingBoxTool  # noqa: F401

