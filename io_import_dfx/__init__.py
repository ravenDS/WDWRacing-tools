# SPDX-FileCopyrightText: 2026 ravenDS
# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "Walt Disney World Quest DFX/DRM Importer",
    "author": "ravenDS",
    "version": (3, 1, 0),
    "blender": (3, 0, 0),
    "location": "File > Import > DFX/DRM Level",
    "description": "Import level geometry and object models from Walt Disney World Quest: Magical Racing Tour DFX/VD3 (PC) and DRM/VRM (PS1) files",
    "category": "Import-Export",
}

import bpy
from bpy.props import StringProperty, BoolProperty, FloatProperty
from bpy_extras.io_utils import ImportHelper


class IMPORT_OT_dfx(bpy.types.Operator, ImportHelper):
    """Import Walt Disney World Quest DFX level (PC)"""
    bl_idname = "import_scene.dfx"
    bl_label = "Import DFX Level (PC)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".dfx"
    filter_glob: StringProperty(default="*.dfx;*.DFX", options={'HIDDEN'})

    vd3_path: StringProperty(name="VD3 Texture File",
        description="Path to matching VD3 texture archive. Leave blank to auto-detect", default="")
    import_level: BoolProperty(name="Import Level Geometry", default=True)
    import_objects: BoolProperty(name="Import Object Models", default=True)
    import_textures: BoolProperty(name="Import Textures", default=True)
    import_overlays: BoolProperty(name="Import Overlay Surfaces", default=True)
    import_animations: BoolProperty(name="Import Bone Animations", default=True,
        description="Import skeletal animations as actions (first action active, others as muted NLA tracks)")
    scale: FloatProperty(name="Scale", default=0.01, min=0.0001, max=100.0)

    def execute(self, context):
        from . import dfx_import
        return dfx_import.load_dfx(context, filepath=self.filepath,
            vd3_path=self.vd3_path, import_level=self.import_level,
            import_objects=self.import_objects, import_textures=self.import_textures,
            import_overlays=self.import_overlays, import_animations=self.import_animations,
            scale=self.scale)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "vd3_path")
        layout.separator()
        layout.prop(self, "import_level")
        layout.prop(self, "import_objects")
        layout.prop(self, "import_textures")
        layout.prop(self, "import_overlays")
        layout.prop(self, "import_animations")
        layout.prop(self, "scale")


class IMPORT_OT_drm(bpy.types.Operator, ImportHelper):
    """Import Walt Disney World Quest DRM level (PS1)"""
    bl_idname = "import_scene.drm"
    bl_label = "Import DRM Level (PS1)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".drm"
    filter_glob: StringProperty(default="*.drm;*.DRM", options={'HIDDEN'})

    vrm_path: StringProperty(name="VRM Texture File",
        description="Path to matching VRM (PS1 VRAM dump). Leave blank to auto-detect", default="")
    import_level: BoolProperty(name="Import Level Geometry", default=True)
    import_objects: BoolProperty(name="Import Object Models", default=True)
    import_textures: BoolProperty(name="Import Textures", default=True)
    import_overlays: BoolProperty(name="Import Overlay Surfaces", default=True)
    import_animations: BoolProperty(name="Import Bone Animations", default=True,
        description="Import skeletal animations as actions (first action active, others as muted NLA tracks)")
    scale: FloatProperty(name="Scale", default=0.01, min=0.0001, max=100.0)

    def execute(self, context):
        from . import dfx_import
        return dfx_import.load_drm(context, filepath=self.filepath,
            vrm_path=self.vrm_path, import_level=self.import_level,
            import_objects=self.import_objects, import_textures=self.import_textures,
            import_overlays=self.import_overlays, import_animations=self.import_animations,
            scale=self.scale)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "vrm_path")
        layout.separator()
        layout.prop(self, "import_level")
        layout.prop(self, "import_objects")
        layout.prop(self, "import_textures")
        layout.prop(self, "import_overlays")
        layout.prop(self, "import_animations")
        layout.prop(self, "scale")


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_dfx.bl_idname, text="WDWR DFX Level - PC (.dfx)")
    self.layout.operator(IMPORT_OT_drm.bl_idname, text="WDWR DRM Level - PS1 (.drm)")


def register():
    bpy.utils.register_class(IMPORT_OT_dfx)
    bpy.utils.register_class(IMPORT_OT_drm)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.utils.unregister_class(IMPORT_OT_drm)
    bpy.utils.unregister_class(IMPORT_OT_dfx)


if __name__ == "__main__":
    register()
