"""Blender script to render images of 3D models. Credits: Objaverse-XL"""

import argparse
import json
import math
import os
import random
import sys
import glob
import json
from time import sleep
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple
from math import radians, degrees

import bpy
import numpy as np
import imageio.v3 as iio
from mathutils import Matrix, Vector

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.open_mainfile,
}

def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    bpy.context.scene.camera = new_camera

def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    """Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def randomize_camera(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 2.2,
    minz: float = -2.2,
    only_northern_hemisphere: bool = False,
) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """

    x, y, z = _sample_spherical(
        radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
    )
    camera = bpy.data.objects["Camera"]

    # only positive z
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector(np.array([x, y, z]))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object

def load_envmap(hdr_path, scale=1):
    # https://github.com/DLR-RM/BlenderProc/issues/2
    bpy.data.worlds.new("World")
    bpy.context.scene.world = bpy.data.worlds.get("World")
    bpy.context.scene.world.use_nodes = True
    bpy.context.scene.world.node_tree.nodes.clear()
    back_node = bpy.context.scene.world.node_tree.nodes.new(type='ShaderNodeBackground')
    node_output = bpy.context.scene.world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
    env_node = bpy.context.scene.world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    texcoord_node = bpy.context.scene.world.node_tree.nodes.new(type="ShaderNodeTexCoord")
    mapping_node = bpy.context.scene.world.node_tree.nodes.new(type="ShaderNodeMapping")
    back_node = bpy.context.scene.world.node_tree.nodes['Background']
    bpy.context.scene.world.node_tree.links.new(env_node.outputs['Color'], back_node.inputs['Color'])
    bpy.context.scene.world.node_tree.links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    bpy.context.scene.world.node_tree.links.new(mapping_node.outputs["Vector"], env_node.inputs["Vector"])
    back_node.inputs[1].default_value = scale #change strength of lighting
    mapping_node.inputs["Rotation"].default_value[2] = np.random.uniform(low=0.0, high=np.pi*2) #random rotate
    bpy.context.scene.world.node_tree.nodes['Environment Texture'].image = bpy.data.images.load(os.path.relpath(hdr_path))
    bpy.context.scene.world.node_tree.links.new(back_node.outputs["Background"], node_output.inputs["Surface"])

def randomize_lighting(location, rotation) -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    light_prob = np.random.uniform()
    if light_prob < 1/2:
        # Create key light
        key_light = _create_light(
            name="Key_Light",
            light_type="POINT",
            location=location,#(0, 0, 0),
            rotation=(0.785398, 0, -0.785398),
            energy=150,
        )
        return "key_light"
    elif 1/2 <= light_prob < 1:
        # Create fill light
        fill_light = _create_light(
            name="Fill_Light",
            light_type="SUN",
            location=location,#(0, 0, 0),
            rotation=rotation,#(0.785398, 0, 2.35619),
            energy=random.choice([10, 15, 20]),
        )
        return "fill_light"

def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        # if obj.type not in {"CAMERA", "LIGHT"}:
        if obj.type not in {"CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    # if not found:
    #     raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)

def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)) or isinstance(obj.data, (bpy.types.Curve)):
            yield obj

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    try:
        for obj in bpy.context.scene.objects:
            if obj.hide_viewport or obj.hide_render:
                obj.hide_viewport = False
                obj.hide_render = False
                obj.hide_select = False
                obj.select_set(True)
    except Exception as e:
        pass
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene() -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    if math.isinf(bbox_max.x):
        scale = 1
    else:
        scale = 1 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    #bbox_min, bbox_max = scene_bbox()
    #offset = -(bbox_min + bbox_max) / 2
    #for obj in get_scene_root_objects():
    #    obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None
    return parent_empty

def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
    obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    rand_color = _get_random_color()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)
    return rand_color


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(
        self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = (
            image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += (
                        len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }

def render_albedo_and_material(idx):
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers[0].use_pass_combined = False
    bpy.context.scene.view_layers[0].use_pass_diffuse_color = True
    bpy.context.scene.view_layers[0].use_pass_normal = True
    bpy.context.scene.view_layers[0].use_pass_z = True
    bpy.context.scene.view_layers[0].cycles.use_denoising = True
    bpy.context.scene.cycles.use_denoising = True

    obj_name = bpy.path.basename(bpy.context.blend_data.filepath).split(".")[0]
    albedo_dir = f"blender_output/throwaway/{obj_name}/{idx}/albedo"
    material_dir = f"blender_output/throwaway/{obj_name}/{idx}/material"
    shaded_dir = f"blender_output/throwaway/{obj_name}/{idx}/shaded"
    normal_dir = f"blender_output/throwaway/{obj_name}/{idx}/normal"
    depth_dir = f"blender_output/throwaway/{obj_name}/{idx}/depth"

    roughness, metalness = 0.0, 0.0
    render_layers = bpy.context.scene.node_tree.nodes.get("Render Layers")
    if render_layers is None:
        render_layers = bpy.context.scene.node_tree.nodes.new("CompositorNodeRLayers")

    if len(bpy.data.objects) > 100 or len(bpy.data.materials) > 100:
        return None, None, None, None, None
    for obj in get_scene_meshes():
        if len(obj.material_slots) == 0:
            mat = bpy.data.materials.new("Material")
            obj.data.materials.append(mat)
        for slot_id, slot in enumerate(obj.material_slots):
            material = slot.material
            if not material:
                material = bpy.data.materials.new("Material")
                obj.material_slots[slot_id].material = material
            if not material.node_tree:
                material.use_nodes = True
            bsdf_node = None
            use_diff_col = False
            for n in material.node_tree.nodes:
                if n.bl_idname == "ShaderNodeBsdfPrincipled":
                    bsdf_node = n
                if n.bl_idname == "ShaderNodeEmission":
                    n.inputs["Strength"].default_value = 0
            if not bsdf_node:
                bsdf_node = material.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
                bsdf_node.inputs["Base Color"].default_value = (0, 0, 0, 1)
                orig_node = None
                for n in material.node_tree.nodes:
                    if n.bl_idname == "ShaderNodeBsdfGlossy":
                        orig_node = n
                if not orig_node:
                    for n in material.node_tree.nodes:
                        if n.bl_idname == "ShaderNodeBsdfDiffuse":
                            orig_node = n
                    if not orig_node:
                        for n in material.node_tree.nodes:
                            if n.bl_idname == "ShaderNodeOutputMaterial":                        
                                material.node_tree.links.new(bsdf_node.outputs["BSDF"], n.inputs[0])
                        bsdf_node.inputs["Roughness"].default_value = 0.25
                    else:
                        bsdf_node.inputs["Roughness"].default_value = 0.25
                        if len(orig_node.inputs["Color"].links) > 0:
                            color_output = orig_node.inputs["Color"].links[0].from_socket
                            material.node_tree.links.new(color_output, bsdf_node.inputs["Base Color"])
                        else:
                            bsdf_node.inputs["Base Color"].default_value = orig_node.inputs["Color"].default_value
                else:
                    bsdf_node.inputs["Roughness"].default_value = orig_node.inputs["Roughness"].default_value
                    bsdf_node.inputs["Metallic"].default_value = 1 - orig_node.inputs["Roughness"].default_value
                    if len(orig_node.inputs["Color"].links) > 0:
                        color_output = orig_node.inputs["Color"].links[0].from_socket
                        material.node_tree.links.new(color_output, bsdf_node.inputs["Base Color"])
                    else:
                        bsdf_node.inputs["Base Color"].default_value = orig_node.inputs["Color"].default_value
            if "Specular" in bsdf_node.inputs:
                bsdf_node.inputs["Specular"].default_value = 0
                if len(bsdf_node.inputs["Specular"].links) > 0:
                    material.node_tree.links.remove(bsdf_node.inputs["Specular"].links[0])
            if "Transmission" in bsdf_node.inputs:
                if bsdf_node.inputs["Transmission"].default_value > 0:
                    bsdf_node.inputs["Base Color"].default_value = (0, 0, 0, 1)
                bsdf_node.inputs["Transmission"].default_value = 0
                if len(bsdf_node.inputs["Transmission"].links) > 0:
                    material.node_tree.links.remove(bsdf_node.inputs["Transmission"].links[0])
            if "Transmission Weight" in bsdf_node.inputs:
                if bsdf_node.inputs["Transmission Weight"].default_value > 0:
                    bsdf_node.inputs["Base Color"].default_value = (0, 0, 0, 1)
                bsdf_node.inputs["Transmission Weight"].default_value = 0
                if len(bsdf_node.inputs["Transmission Weight"].links) > 0:
                    material.node_tree.links.remove(bsdf_node.inputs["Transmission Weight"].links[0])
            if "Sheen" in bsdf_node.inputs:
                bsdf_node.inputs["Sheen"].default_value = 0
            if "Subsurface" in bsdf_node.inputs:
                bsdf_node.inputs["Subsurface"].default_value = 0
            if "Clearcoat" in bsdf_node.inputs:
                bsdf_node.inputs["Clearcoat"].default_value = 0
            if "Emission" in bsdf_node.inputs:
                bsdf_node.inputs["Emission"].default_value = (0, 0, 0, 1)
            for n in material.node_tree.nodes:
                if n.bl_idname == "ShaderNodeBsdfPrincipled":
                    if "Specular" in n.inputs:
                        n.inputs["Specular"].default_value = 0
                        if len(n.inputs["Specular"].links) > 0:
                            material.node_tree.links.remove(n.inputs["Specular"].links[0])
                    if "Transmission" in n.inputs:
                        if n.inputs["Transmission"].default_value > 0:
                            n.inputs["Base Color"].default_value = (0, 0, 0, 1)
                        n.inputs["Transmission"].default_value = 0
                        if len(n.inputs["Transmission"].links) > 0:
                            material.node_tree.links.remove(n.inputs["Transmission"].links[0])
                    if "Transmission Weight" in n.inputs:
                        if n.inputs["Transmission Weight"].default_value > 0:
                            n.inputs["Base Color"].default_value = (0, 0, 0, 1)
                        n.inputs["Transmission Weight"].default_value = 0
                        if len(n.inputs["Transmission Weight"].links) > 0:
                            material.node_tree.links.remove(n.inputs["Transmission Weight"].links[0])
                    if "Sheen" in n.inputs:
                        n.inputs["Sheen"].default_value = 0
                    if "Subsurface" in n.inputs:
                        n.inputs["Subsurface"].default_value = 0
                    if "Clearcoat" in n.inputs:
                        n.inputs["Clearcoat"].default_value = 0
                    if "Emission" in n.inputs:
                        n.inputs["Emission"].default_value = (0, 0, 0, 1)
            file_node_albedo = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
            file_node_orm = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
            file_node_img = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
            alpha_set_albedo = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
            alpha_set_orm = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
            alpha_set_img = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
            file_node_albedo.base_path = albedo_dir
            file_node_orm.base_path = material_dir
            file_node_img.base_path = shaded_dir

            aov_orm = material.node_tree.nodes.new(type="ShaderNodeOutputAOV")
            aov_orm.name = "ORM"
            aov_albedo = material.node_tree.nodes.new(type="ShaderNodeOutputAOV")
            aov_albedo.name = "Albedo"
            combine_node = material.node_tree.nodes.new(type="ShaderNodeCombineColor")
            try:
                metallic_connected_output = bsdf_node.inputs["Metallic"].links[0].from_socket
                material.node_tree.links.new(metallic_connected_output, combine_node.inputs["Blue"])
            except IndexError:
                metalness = bsdf_node.inputs["Metallic"].default_value
                combine_node.inputs["Blue"].default_value = metalness
            try:
                roughness_connected_output = bsdf_node.inputs["Roughness"].links[0].from_socket
                material.node_tree.links.new(roughness_connected_output, combine_node.inputs["Green"])
            except IndexError:
                roughness = bsdf_node.inputs["Roughness"].default_value
                combine_node.inputs["Green"].default_value = roughness
            try:
                albedo_connected_output = bsdf_node.inputs["Base Color"].links[0].from_socket
                material.node_tree.links.new(albedo_connected_output, aov_albedo.inputs["Color"])
            except IndexError:
                aov_albedo.inputs["Color"].default_value = bsdf_node.inputs["Base Color"].default_value
            material.node_tree.links.new(combine_node.outputs["Color"], aov_orm.inputs["Color"])
            bpy.ops.scene.view_layer_add_aov()
            bpy.ops.scene.view_layer_add_aov()
            bpy.context.view_layer.aovs[0].name = "ORM"
            bpy.context.view_layer.aovs[1].name = "Albedo"
            if use_diff_col:
                bpy.context.scene.node_tree.links.new(render_layers.outputs["DiffCol"], alpha_set_albedo.inputs["Image"])
            else:
                bpy.context.scene.node_tree.links.new(render_layers.outputs["Albedo"], alpha_set_albedo.inputs["Image"])
            bpy.context.scene.node_tree.links.new(render_layers.outputs["Alpha"], alpha_set_albedo.inputs["Alpha"])
            bpy.context.scene.node_tree.links.new(alpha_set_albedo.outputs["Image"], file_node_albedo.inputs[0])
            bpy.context.scene.node_tree.links.new(render_layers.outputs["ORM"], alpha_set_orm.inputs["Image"])
            bpy.context.scene.node_tree.links.new(render_layers.outputs["Alpha"], alpha_set_orm.inputs["Alpha"])
            bpy.context.scene.node_tree.links.new(alpha_set_orm.outputs["Image"], file_node_orm.inputs[0])
            bpy.context.scene.node_tree.links.new(render_layers.outputs["Image"], alpha_set_img.inputs["Image"])
            bpy.context.scene.node_tree.links.new(render_layers.outputs["Alpha"], alpha_set_img.inputs["Alpha"])
            bpy.context.scene.node_tree.links.new(alpha_set_img.outputs["Image"], file_node_img.inputs[0])

    bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
    file_node = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
    alpha_set = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
    file_node.base_path = normal_dir
    bpy.context.scene.node_tree.links.new(render_layers.outputs["Normal"], alpha_set.inputs["Image"])
    bpy.context.scene.node_tree.links.new(render_layers.outputs["Alpha"], alpha_set.inputs["Alpha"])
    bpy.context.scene.node_tree.links.new(alpha_set.outputs["Image"], file_node.inputs[0])

    file_node_depth = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
    alpha_set_depth = bpy.context.scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
    file_node_depth.base_path = depth_dir
    bpy.context.scene.node_tree.links.new(render_layers.outputs["Depth"], alpha_set_depth.inputs["Image"])
    bpy.context.scene.node_tree.links.new(render_layers.outputs["Alpha"], alpha_set_depth.inputs["Alpha"])
    bpy.context.scene.node_tree.links.new(alpha_set_depth.outputs["Image"], file_node_depth.inputs[0])
    bpy.ops.render.render(write_still=True)
    from time import sleep

    rendered_img, iter = None, 0
    while rendered_img is None:
        if iter > 2:
            return None, None, None, None, None
        try:
            render_imgs = glob.glob(shaded_dir + "/*.png")
            rendered_img = iio.imread(render_imgs[0])
            albedo_imgs = glob.glob(albedo_dir + "/*.png")
            albedo = iio.imread(albedo_imgs[0])
            orm_imgs = glob.glob(material_dir + "/*.png")
            orm = iio.imread(orm_imgs[0])
            depth_imgs = glob.glob(depth_dir + "/*.exr")
            depth = iio.imread(depth_imgs[0])
            normal_imgs = glob.glob(normal_dir + "/*.exr")
            normal = iio.imread(normal_imgs[0])
            # normal = -iio.imread(normal_imgs[0]) #uncomment if using denoising normal to get camera space normals
        except Exception as err:
            bpy.ops.render.render(write_still=True)
            iter += 1
    normal = normal / 2 + 0.5
    alpha = albedo[..., 3:]
    normal[..., 3:] = alpha
    normal = normal.clip(0, 1)
    from PIL import Image
    normal = Image.fromarray((normal * 255).astype(np.uint8))
    # alpha = np.squeeze(alpha, axis=2)
    # alpha = np.where(alpha >= 127, 255, 0).astype(np.uint8)
    
    #cleanup compositor
    tree = bpy.context.scene.node_tree
    for node in tree.nodes:
        if node != render_layers:
            tree.nodes.remove(node)

    return rendered_img, albedo, orm, normal, depth#, alpha

def render_object_single_image(
    object_file: str,
    obj_idx: int,
    envmap: str,
    only_northern_hemisphere: bool,
    mode: str,
    step: float = -1.0,
    scale: float = 100.0,
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    use_background: bool = False
):
    bpy.ops.wm.read_homefile()
    # load the object
    file_extension = object_file.split(".")[-1].lower()
    import_function = IMPORT_FUNCTIONS[file_extension]
    if file_extension == "blend":
        import_function(filepath=object_file)
        reset_cameras()
        delete_invisible_objects()
    elif file_extension in {"glb", "gltf"}:
        reset_scene()
        import_function(filepath=object_file, merge_vertices=True)
    else:
        raise Exception("Invalid object file type!")

    # Set up cameras
    cam = bpy.context.scene.objects["Camera"]

    # Set up camera constraints
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    empty = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty)

    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    if math.isinf(bbox_max[0]):
        return None, None, None, None, None, None, None
    center = (bbox_max + bbox_min) / 2
    empty.location = center
    bpy.ops.object.select_all(action="DESELECT")
    if mode == "nerf":
        cam.parent = empty  # setup parenting
        bpy.context.view_layer.objects.active = empty
    cam_constraint.target = empty

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.cycles.adaptive_threshold = 0.1
    bpy.context.scene.cycles.use_fast_gi = True
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.film_transparent = use_background

    if scale > 0 and envmap:
        load_envmap(envmap, scale=scale)

    # normalize the scene
    normalize_scene()

    # render the images
    # set camera
    if mode == "zero123":
        camera = randomize_camera(
            radius_min=radius_min,
            radius_max=radius_max,
            only_northern_hemisphere=only_northern_hemisphere,
        )
        rt_matrix = get_3x4_RT_matrix_from_blender(camera)
    elif mode == "nerf":
        if step >= 0:
            x = radius_min * np.cos(np.pi/2 + step * np.pi/180)
            y = radius_min * np.sin(np.pi/2 + step * np.pi/180)
            z = 1.8
        else:
            elevation = np.random.uniform(0, 90 + 15) * np.pi/180
            azimuth = np.random.uniform(0, 360) * np.pi/180
            radius = np.random.uniform(radius_min, radius_max)
            x = radius * np.sin(elevation) * np.cos(azimuth)
            y = radius * np.sin(elevation) * np.sin(azimuth)
            z = radius * np.cos(elevation)
        if only_northern_hemisphere:
            z = abs(z)
        cam.location = Vector(np.array([x, y, z]))
        cam.update_tag()
        bpy.context.view_layer.update() #To update the cam.matrix_world
        if scale < 0:
            location, rotation = cam.matrix_world.decompose()[0:2]
            envmap = randomize_lighting(cam.location, rotation.to_euler())

        if mode == "nerf":
            rt_matrix = {
                "envmap": envmap,
                'transform_matrix': listify_matrix(cam.matrix_world)
            }

    else:
        raise ValueError(f"Invalid mode: {mode}")

    # render the image
    bpy.context.scene.render.filepath = "blender_output/blender_img.png"

    rendered_img, albedo, orm, normal, depth = render_albedo_and_material(obj_idx)

    return rendered_img, albedo, orm, normal, depth, None, rt_matrix

def save_transforms(args, out_data, out_data_test):
    if args.mode == "zero123":
        with open(os.path.join(args.output_dir, "envmap_mappings.json"), "w") as f:
            json.dump(out_data, f, indent=4)
        with open(os.path.join(test_output_dir, "envmap_mappings_test.json"), "w") as f:
            json.dump(out_data, f, indent=4)
    elif args.mode == "nerf":
        out_data['camera_angle_x'] = bpy.data.objects['Camera'].data.angle_x
        out_data_test['camera_angle_x'] = bpy.data.objects['Camera'].data.angle_x
        with open(os.path.join(args.output_dir, "transforms_train.json"), "w") as f:
            json.dump(out_data, f)
        with open(os.path.join(args.output_dir, "transforms_test.json"), "w") as f:
            json.dump(out_data_test, f)
        with open(os.path.join(args.output_dir, "transforms_val.json"), "w") as f:
            json.dump(out_data_test, f)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objects_path",
        type=str,
        required=True,
        help="Path to the object files",
    )
    parser.add_argument(
        "--envmaps_path",
        type=str,
        default="data/irrmaps/fullres_light_probes",
        help="Path to environment maps"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="blender_output/blender_renders",
        help="Output directory"
    )
    parser.add_argument(
        "--only_northern_hemisphere",
        action="store_true",
        help="Only render the northern hemisphere of the object (only applies to zero123 rendering).",
        default=False,
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=12,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--radius_min",
        type=float,
        default=1.6,
        help="Minimum distance from object.",
    )
    parser.add_argument(
        "--radius_max",
        type=float,
        default=2.1,
        help="Maximum distance from object.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="nerf",
        help="Pose output format [zero123, nerf].",
    )
    parser.add_argument(
        "--render_panorama",
        action="store_true",
        help="Render panorama poses of the object.",
        default=False,
    )
    parser.add_argument(
        "--resume_rendering",
        action="store_true",
        help="Resume rendering obejcts from last time.",
        default=False,
    )
    parser.add_argument(
        "--begin_idx",
        type=int,
        default=0,
        help="Object index to begin to rendering.",
    )
    parser.add_argument(
        "--use_background",
        action="store_true",
        help="Render envmap background in the images.",
        default=False,
    )
    args = parser.parse_args()

    context = bpy.context
    scene = bpy.context.scene
    render = scene.render

    # Set cycles settings
    scene.cycles.device = "GPU"
    scene.cycles.samples = 16
    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transparent_max_bounces = 1
    scene.cycles.transmission_bounces = 1
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "OPTIX"

    objects = glob.glob(os.path.join(args.objects_path, "**/*.blend"), recursive=True)
    if len(objects) == 0:
        objects = glob.glob(os.path.join(args.objects_path, "**/*.glb"), recursive=True) + glob.glob(os.path.join(args.objects_path, "**/*.gltf"), recursive=True)
    if len(objects) == 0:
        raise ValueError("No objects found in the given directory")
    print("Rendering {} images, {} objects".format(args.num_renders * len(objects), len(objects)))
    envlights = glob.glob(args.envmaps_path + "/*.hdr") + glob.glob(args.envmaps_path + "/*.exr")
    train_output_dir = os.path.join(args.output_dir, "train")
    test_output_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Render the images
    if os.path.exists(os.path.join(args.output_dir, "envmap_mappings.json")) and args.resume_rendering:
        with open(os.path.join(args.output_dir, "envmap_mappings.json"), "r") as f:
            out_data = json.load(f)
        with open(os.path.join(args.output_dir, "envmap_mappings_test.json"), "r") as f:
            out_data_test = json.load(f)
        if args.begin_idx <= 0:
            args.begin_idx = int(out_data_test.keys()[-1].split('/')[1])
    elif os.path.exists(os.path.join(args.output_dir, "transforms_train.json")) and args.resume_rendering:
        with open(os.path.join(args.output_dir, "transforms_train.json"), "r") as f:
            out_data = json.load(f)
        with open(os.path.join(args.output_dir, "transforms_test.json"), "r") as f:
            out_data_test = json.load(f)
        if args.begin_idx <= 0:
            args.begin_idx = int(out_data['frames'][-1]['file_path'].split('/')[1])
    else:
        out_data, out_data_test = {}, {}
        out_data['frames'], out_data_test['frames'] = [], []
    if not args.resume_rendering:
        out_data, out_data_test = {}, {}
        out_data['frames'], out_data_test['frames'] = [], []
    stepsize = 360 / args.num_renders
    for idx, obj in enumerate(objects[args.begin_idx:], args.begin_idx):
        step = 0.0
        if idx % 10 == 0:
            save_transforms(args, out_data, out_data_test)
        if os.path.getsize(obj) > 5e8:
            print("Skipping large object {}".format(obj))
            continue
        for i in range(args.num_renders):
            light_prob = np.random.uniform()
            if light_prob <= 1/2:
                envmap = random.choice(envlights)
                scale = 1
            else:
                envmap = None
                scale = -1
            print("Rendering object {}, idx: {}, iteration: {}".format(obj, idx, i))
            render_test = idx >= int(len(objects) * 0.99)
            render_img, albedo, orm, normal, depth, alpha, rt_matrix = render_object_single_image(
                object_file=obj,
                envmap=envmap,
                only_northern_hemisphere=args.only_northern_hemisphere,
                mode=args.mode,
                scale=scale,
                step=step if args.render_panorama and not render_test else -1.0,
                radius_min=args.radius_max if args.render_panorama and not render_test else args.radius_min,
                radius_max=args.radius_max if args.render_panorama and not render_test else args.radius_max,
                obj_idx=idx,
                use_background=not args.use_background
            )
            if render_img is not None:
                if render_test:
                    os.makedirs(os.path.join(test_output_dir, f"{idx:05}"), exist_ok=True)
                    iio.imwrite(os.path.join(test_output_dir, f"{idx:05}/{i:03}.png"), render_img)
                    iio.imwrite(os.path.join(test_output_dir, f"{idx:05}/{i:03}_albedo.png"), albedo)
                    iio.imwrite(os.path.join(test_output_dir, f"{idx:05}/{i:03}_orm.png"), orm)
                    iio.imwrite(os.path.join(test_output_dir, f"{idx:05}/{i:03}_normal.png"), normal)
                    iio.imwrite(os.path.join(test_output_dir, f"{idx:05}/{i:03}_depth.exr"), depth)
                    if args.mode == "zero123":
                        np.save(os.path.join(test_output_dir, f"{i:05}.npy"), rt_matrix)
                        out_data_test[f"test/{idx:05}/{i:05}.png"] = envmap
                    elif args.mode == "nerf":
                        rt_matrix['file_path'] = f"test/{idx:05}/{i:03}"
                        rt_matrix['file_path_normal'] = f"test/{idx:05}/{i:03}_normal"
                        rt_matrix['file_path_albedo'] = f"test/{idx:05}/{i:03}_albedo"
                        rt_matrix['file_path_orm'] = f"test/{idx:05}/{i:03}_orm"
                        rt_matrix['file_path_depth'] = f"test/{idx:05}/{i:03}_depth"
                        out_data_test['frames'].append(rt_matrix)
                else:
                    os.makedirs(os.path.join(train_output_dir, f"{idx:05}"), exist_ok=True)
                    iio.imwrite(os.path.join(train_output_dir, f"{idx:05}/{i:03}.png"), render_img)
                    iio.imwrite(os.path.join(train_output_dir, f"{idx:05}/{i:03}_albedo.png"), albedo)
                    iio.imwrite(os.path.join(train_output_dir, f"{idx:05}/{i:03}_orm.png"), orm)
                    iio.imwrite(os.path.join(train_output_dir, f"{idx:05}/{i:03}_normal.png"), normal)
                    iio.imwrite(os.path.join(train_output_dir, f"{idx:05}/{i:03}_depth.exr"), depth)
                    if args.mode == "zero123":
                        np.save(os.path.join(train_output_dir, f"{idx:05}/{i:03}.npy"), rt_matrix)
                        out_data[f"train/{idx:05}/{i:05}.png"] = envmap
                    elif args.mode == "nerf":
                        rt_matrix['file_path'] = f"train/{idx:05}/{i:03}"
                        rt_matrix['file_path_albedo'] = f"train/{idx:05}/{i:03}_albedo"
                        rt_matrix['file_path_normal'] = f"train/{idx:05}/{i:03}_normal"
                        rt_matrix['file_path_orm'] = f"train/{idx:05}/{i:03}_orm"
                        rt_matrix['file_path_depth'] = f"train/{idx:05}/{i:03}_depth"
                        out_data['frames'].append(rt_matrix)
            else:
                break
            step += stepsize
    save_transforms(args, out_data, out_data_test)