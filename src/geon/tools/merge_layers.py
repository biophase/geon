"""Undoable replacement of whole layers by a same-type merged layer."""
from .command_manager import Command


class MergeLayersCommand(Command):
    def __init__(self, scene, sources, merged, refresh):
        super().__init__('Merge layers')
        self.scene=scene
        self.sources=list(sources)
        self.merged=merged
        self.refresh=refresh
        self.order=list(scene.layers)
        self.previous_active=scene.active_layer_id

    def _add(self, layer):
        self.scene.doc.add_data(layer.data)
        try:
            self.scene.add_layer(layer)
        except Exception:
            if layer.id in self.scene.layers:
                self.scene.remove_layer(layer.id,delete_data=False)
            self.scene.doc.remove_data(layer.id)
            raise

    def _reorder(self, order):
        current=self.scene._layers
        keys=[i for i in order if i in current]+[i for i in current if i not in order]
        self.scene._layers={i:current[i] for i in keys}
        data=self.scene.doc.scene_items
        self.scene.doc.scene_items={i:data[i] for i in keys if i in data} | {i:v for i,v in data.items() if i not in keys}

    def execute(self):
        if any(self.scene.layers.get(layer.id) is not layer for layer in self.sources):
            raise ValueError('The source layers changed before merging.')
        # Attach first: a rendering failure must leave every source intact.
        self._add(self.merged)
        ids={layer.id for layer in self.sources}
        for layer in self.sources:self.scene.remove_layer(layer.id)
        order=[];inserted=False
        for key in self.order:
            if key in ids:
                if not inserted:order.append(self.merged.id);inserted=True
            else:order.append(key)
        self._reorder(order)
        self.scene.active_layer_id=self.merged.id
        self.refresh()

    def undo(self):
        added=[]
        try:
            for layer in self.sources:
                self._add(layer);added.append(layer)
        except Exception:
            for layer in added:self.scene.remove_layer(layer.id)
            raise
        self.scene.remove_layer(self.merged.id)
        self._reorder(self.order)
        self.scene.active_layer_id=self.previous_active if self.previous_active in self.scene.layers else self.sources[0].id
        self.refresh()
