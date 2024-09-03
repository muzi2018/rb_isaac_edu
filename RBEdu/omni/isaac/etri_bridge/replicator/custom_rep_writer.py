from typing import List

from omni.replicator.core import BasicWriter
from omni.syntheticdata.scripts.SyntheticData import SyntheticData
from omni.replicator.core.bindings._omni_replicator_core import Schema_omni_replicator_extinfo_1_0

import numpy as np
import omni.replicator.core as rep
from omni.replicator.core import functional as F
from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch


__version__ = "0.0.1"

def colorize_normals(data):
    """Convert normals data into colored image.

    Args:
        data (numpy.ndarray): data returned by the annotator.

    Return:
        (numpy.ndarray): Data converted to uint8 RGB image.
    """

    colored_data = ((data * 0.5 + 0.5) * 255).astype(np.uint8)
    return colored_data

class CustomBasicWriter(BasicWriter):

    def __init__(
        self,
        output_dir: str,
        s3_bucket: str = None,
        s3_region: str = None,
        s3_endpoint: str = None,
        semantic_types: List[str] = None,
        rgb: bool = False,
        bounding_box_2d_tight: bool = False,
        bounding_box_2d_loose: bool = False,
        semantic_segmentation: bool = False,
        instance_id_segmentation: bool = False,
        instance_segmentation: bool = False,
        # background_rand: bool = False,
        distance_to_camera: bool = False,
        distance_to_image_plane: bool = False,
        bounding_box_3d: bool = False,
        occlusion: bool = False,
        normals: bool = False,
        motion_vectors: bool = False,
        camera_params: bool = False,
        pointcloud: bool = False,
        pointcloud_include_unlabelled: bool = False,
        image_output_format: str = "png",
        colorize_semantic_segmentation: bool = True,
        colorize_instance_id_segmentation: bool = True,
        colorize_instance_segmentation: bool = True,
        skeleton_data: bool = False,
        frame_padding: int = 4,
        semantic_filter_predicate: str = None,
        frame_rate: int = 1,
    ):
        super().__init__(
            output_dir=output_dir,
            s3_bucket=s3_bucket,
            s3_region=s3_region,
            s3_endpoint=s3_endpoint,
            semantic_types=semantic_types,
            rgb=rgb,
            bounding_box_2d_tight=bounding_box_2d_tight,
            bounding_box_2d_loose=bounding_box_2d_loose,
            semantic_segmentation=semantic_segmentation,
            instance_id_segmentation=instance_id_segmentation,
            instance_segmentation=instance_segmentation,
            # background_rand=background_rand,
            distance_to_camera=distance_to_camera,
            distance_to_image_plane=distance_to_image_plane,
            bounding_box_3d=bounding_box_3d,
            occlusion=occlusion,
            normals=normals,
            motion_vectors=motion_vectors,
            camera_params=camera_params,
            pointcloud=pointcloud,
            pointcloud_include_unlabelled=pointcloud_include_unlabelled,
            image_output_format=image_output_format,
            colorize_semantic_segmentation=colorize_semantic_segmentation,
            colorize_instance_id_segmentation=colorize_instance_id_segmentation,
            colorize_instance_segmentation=colorize_instance_segmentation,
            skeleton_data=skeleton_data,
            frame_padding=frame_padding,
            semantic_filter_predicate=semantic_filter_predicate
        )
        self._frame_rate = frame_rate
        self._frame_id = 0
        self._capture_id = 0
        self._sequence_id = 0
        

    def write(
            self, 
            data: dict
        ):
        """Write function called from the OgnWriter node on every frame to process annotator output.

        Args:
            data: A dictionary containing the annotator data for the current frame.
        """
        # Check for on_time triggers
        # For each on_time trigger, prefix the output frame number with the trigger counts
        sequence_id = ""
        for trigger_name, call_count in data["trigger_outputs"].items():
            if "on_time" in trigger_name:
                sequence_id = f"{call_count}_{sequence_id}"
        if sequence_id != self._sequence_id:
            self._frame_id = 0
            self._sequence_id = sequence_id

        # print(
        #     f"self._frame_id: {self._frame_id} sequence_id: {self._sequence_id} capture_id: {self._capture_id}"
        # )

        # self._frame_id += self._frame_rate
        if self._capture_id % self._frame_rate == 0:
            for annotator in data.keys():
                annotator_split = annotator.split("-")
                render_product_path = ""
                multi_render_prod = 0
                # multiple render_products
                if len(annotator_split) > 1:
                    multi_render_prod = 1
                    render_product_name = annotator_split[-1]
                    render_product_path = f"{render_product_name}/"

                if annotator.startswith("rgb"):
                    # if multi_render_prod:
                    render_product_path += "rgb/"
                    self._write_rgb(data, render_product_path, annotator)

                if annotator.startswith("Aug"):
                    if isinstance(data[annotator], dict):
                        data[annotator] = data[annotator]["data"]
                    # if multi_render_prod:
                    render_product_path += f"{annotator}/"
                    self._write_rgb(data, render_product_path, annotator)

                if annotator.startswith("normals"):
                    # if multi_render_prod:
                    render_product_path += "normals/"
                    self._write_normals(data, render_product_path, annotator)

                if annotator.startswith("distance_to_camera"):
                    # if multi_render_prod:
                    render_product_path += "distance_to_camera/"
                    self._write_distance_to_camera(data, render_product_path, annotator)

                if annotator.startswith("distance_to_image_plane"):
                    # if multi_render_prod:
                    render_product_path += "distance_to_image_plane/"
                    self._write_distance_to_image_plane(data, render_product_path, annotator)

                if annotator.startswith("semantic_segmentation"):
                    # if multi_render_prod:
                    render_product_path += "semantic_segmentation/"
                    self._write_semantic_segmentation(data, render_product_path, annotator)

                if annotator.startswith("instance_id_segmentation"):
                    # if multi_render_prod:
                    render_product_path += "instance_id_segmentation/"
                    self._write_instance_id_segmentation(data, render_product_path, annotator)

                if annotator.startswith("instance_segmentation"):
                    # if multi_render_prod:
                    render_product_path += "instance_segmentation/"
                    self._write_instance_segmentation(data, render_product_path, annotator)

                if annotator.startswith("motion_vectors"):
                    # if multi_render_prod:
                    render_product_path += "motion_vectors/"
                    self._write_motion_vectors(data, render_product_path, annotator)

                if annotator.startswith("occlusion"):
                    # if multi_render_prod:
                    render_product_path += "occlusion/"
                    self._write_occlusion(data, render_product_path, annotator)

                if annotator.startswith("bounding_box_3d"):
                    # if multi_render_prod:
                    render_product_path += "bounding_box_3d/"
                    self._write_bounding_box_data(data, "3d", render_product_path, annotator)

                if annotator.startswith("bounding_box_2d_loose"):
                    # if multi_render_prod:
                    render_product_path += "bounding_box_2d_loose/"
                    self._write_bounding_box_data(data, "2d_loose", render_product_path, annotator)

                if annotator.startswith("bounding_box_2d_tight"):
                    # if multi_render_prod:
                    render_product_path += "bounding_box_2d_tight/"
                    self._write_bounding_box_data(data, "2d_tight", render_product_path, annotator)

                if annotator.startswith("camera_params"):
                    # if multi_render_prod:
                    render_product_path += "camera_params/"
                    self._write_camera_params(data, render_product_path, annotator)

                if annotator.startswith("pointcloud"):
                    # if multi_render_prod:
                    render_product_path += "pointcloud/"
                    self._write_pointcloud(data, render_product_path, annotator)

                if annotator.startswith("skeleton_data"):
                    # if multi_render_prod:
                    render_product_path += "skeleton_data/"
                    self._write_skeleton(data, render_product_path, annotator)

            self._frame_id += 1
            self._capture_id = 0

        self._capture_id += 1


    def _write_pointcloud(
            self, 
            data: dict, 
            render_product_path: str, 
            annotator: str
        ):

        pointcloud_data = data[annotator]["data"]
        # pointcloud_rgb = data[annotator]["info"]["pointRgb"].reshape(-1, 4)
        # pointcloud_normals = data[annotator]["info"]["pointNormals"].reshape(-1, 4)
        # pointcloud_semantic = data[annotator]["info"]["pointSemantic"]
        pointcloud_instance = data[annotator]["info"]["pointInstance"]

        file_path = f"{render_product_path}pointcloud_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.schedule(F.write_np, data=pointcloud_data, path=file_path)

        # rgb_file_path = (
        #     f"{render_product_path}pointcloud_rgb_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        # )
        # self._backend.schedule(F.write_np, data=pointcloud_rgb, path=rgb_file_path)

        # normals_file_path = (
        #     f"{render_product_path}pointcloud_normals_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        # )
        # self._backend.schedule(F.write_np, data=pointcloud_normals, path=normals_file_path)

        # semantic_file_path = (
        #     f"{render_product_path}pointcloud_semantic_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        # )
        # self._backend.schedule(F.write_np, data=pointcloud_semantic, path=semantic_file_path)

        instance_file_path = (
            f"{render_product_path}pointcloud_instance_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.schedule(F.write_np, data=pointcloud_instance, path=instance_file_path)

rep.WriterRegistry.register(CustomBasicWriter)
