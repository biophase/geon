import h5py

from geon.data.document import Document


def test_load_accepts_older_format_version(tmp_path) -> None:
    path = tmp_path / "old_format.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("document")
        group.attrs["type"] = "Document"
        group.attrs["geon_format_version"] = 1
        group.attrs["name"] = "old"

    doc = Document.load_hdf5(path)

    assert doc.name == "old"

