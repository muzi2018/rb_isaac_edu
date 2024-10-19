from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.physx.scripts import physicsUtils, particleUtils
from omni.isaac.sensor import Camera


from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Sdf, Gf, Tf, UsdLux
from semantics.schema.editor import PrimSemanticData
import omni.replicator.core as rep

import numpy as np
import random
import omni
import os


#### Light functions ####

def turn_off_dg_light():
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/defaultGroundPlane/SphereLight")
    attribute = prim.GetAttribute("intensity")
    attribute.Set(0)


def add_random_light(
        min_trans: list = [0.0, 0.0, 0.0],
        max_trans: list = [0.0, 0.0, 0.0],
        min_rotation: list = [0.0, 0.0, 0.0],
        max_rotation: list = [0.0, 0.0, 0.0],
        min_scale: list = [0.0, 0.0, 0.0],
        max_scale: list = [0.0, 0.0, 0.0],
        light_type: str = "sphere",
        min_color: list = [0.0, 0.0, 0.0],
        max_color: list = [0.0, 0.0, 0.0],
        min_intensity: float = 0.0,
        max_intensity: float = 1000.0,
        min_temperature: float = 6500.0,
        max_temperature: float = 6500.0,
    ):
    lights = rep.create.light(
        position=rep.distribution.uniform(min_trans, max_trans),
        rotation=rep.distribution.uniform(min_rotation, max_rotation),
        scale=rep.distribution.uniform(min_scale, max_scale),
        light_type=light_type,
        color=rep.distribution.uniform(min_color, max_color),
        intensity=rep.distribution.uniform(min_intensity, max_intensity),
        temperature=rep.distribution.uniform(min_temperature, max_temperature),
        count=1,
    )
    return lights


#### Camera functions ####

def add_camera(
        prim_path: str,
        position: list = [0.0, 0.0, 10.0],
        frequency: int = 30,
        resolution: tuple = (640, 480),
        orientation: list = [0.0, 0.0, 0.0, 1.0],
        focal_length: float = 1.0,
    ):

    camera = Camera(
        prim_path=prim_path,
        position=np.array(position),
        frequency=30,
        resolution=(640, 480),
        orientation=np.array(orientation),
    )
    # ref: https://forums.developer.nvidia.com/t/orientation-of-camera/260757/2
    camera.set_world_pose(np.array(position), np.array(orientation), camera_axes="usd")
    camera.set_focal_length(focal_length)
    camera.initialize()
    camera.add_motion_vectors_to_frame()
    return


#### Object functions ####

def add_object(
        usd_path: str, 
        prim_path: str,
        translate: tuple = (0.0, 0.0, 0.0),
        orient: tuple = (0.0, 0.0, 0.0, 1.0),
        scale: tuple = (1.0, 1.0, 1.0),
    ) -> None:
    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    obj_mesh = UsdGeom.Mesh.Get(get_current_stage(), prim_path)
    tx, ty, tz = tuple(translate)
    physicsUtils.set_or_add_translate_op(
        obj_mesh,
        translate=Gf.Vec3f(tx, ty, tz),
    )
    rw, rx, ry, rz = tuple(orient)
    physicsUtils.set_or_add_orient_op(
        obj_mesh,
        orient=Gf.Quatf(rw, rx, ry, rz),
    )

    # if scale type if tuple
    if isinstance(scale, tuple):
        sx, sy, sz = tuple(scale)
        physicsUtils.set_or_add_scale_op(
            obj_mesh,
            scale=Gf.Vec3f(sx, sy, sz),
        )
    elif isinstance(scale, float):
        physicsUtils.set_or_add_scale_op(
            obj_mesh,
            scale=Gf.Vec3f(scale, scale, scale),
        )
    return


def add_semantic_data(
        prim_path: str, 
        class_name: str
    ) -> None:
    print(f"[add_semantic_data] prim_path: {prim_path}, class_name: {class_name}")
    prim = get_current_stage().GetPrimAtPath(prim_path)
    prim_sd = PrimSemanticData(prim)
    prim_sd.add_entry("class", class_name)


def add_bg(
        usd_path: str,
        prim_path: str,
        scale, # scale can be float or tuple
        translate: tuple = (0.0, 0.0, 0.0),
        orient: tuple = (1.0, 0.0, 0.0, 0.0),
        collision_approximation: str = "meshSimplification",
    ):

    add_object(
        usd_path=usd_path,
        prim_path=prim_path,
        translate=translate,
        orient=orient,
        scale=scale,
    )
    add_physical_properties(
        prim_path=prim_path,
        rigid_body=False,
        collision_approximation=collision_approximation
    )
    add_semantic_data(prim_path, "background")

    return


def add_random_object_visual(
        usd_path: str, 
        prim_path: str, 
        semantic_class: str,
        # scale can be float or tuple
        min_scale,
        max_scale,
        min_trans: list = [0.0, 0.0, 0.0],
        max_trans: list = [0.0, 0.0, 0.0],
        min_rotation: list = [0.0, 0.0, 0.0],
        max_rotation: list = [0.0, 0.0, 0.0],
    ):

    min_trans, max_trans = np.array(min_trans), np.array(max_trans)
    min_rot, max_rot = np.radians(np.array(min_rotation)), np.radians(np.array(max_rotation))
    
    random_trans = np.random.uniform(min_trans, max_trans)
    random_rot = np.random.uniform(min_rot, max_rot)
    random_scale = np.random.uniform(min_scale, max_scale)

    if isinstance(min_scale, tuple) or isinstance(min_scale, list):
        random_scale_temp = np.random.uniform(min_scale, max_scale)
        scale_x, scale_y, scale_z = random_scale_temp
        random_scale = (scale_x, scale_y, scale_z)
    elif isinstance(min_scale, float):
        random_scale = np.random.uniform(min_scale, max_scale)
    
    add_object(
        usd_path=usd_path,
        prim_path=prim_path,
        translate=random_trans,
        orient=euler_angles_to_quat(random_rot),
        scale=random_scale,
    )
    add_semantic_data(prim_path, semantic_class)
    return random_trans, euler_angles_to_quat(random_rot), random_scale


#### Physics functions ####

def add_random_object_port_physics(
        world,
        prim_path, 
        ini_pos,
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        scale=np.array([1.0, 1.0, 1.0]),
        static_friction=None,
        dynamic_friction=None,
        kinematic_enabled=False,
        collision_type=None, 
        mass_value=None,
    ):

    bodyPrim = world.stage.GetPrimAtPath(prim_path)
    meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(bodyPrim)
    meshCollision.CreateSdfResolutionAttr().Set(350)
    
    # RigidBody
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(bodyPrim)
    if kinematic_enabled:
        rigid_api.CreateKinematicEnabledAttr(True)

    geom_name = prim_path.split("/")[-2] + "_" + prim_path.split("/")[-1]
    geom = GeometryPrim(prim_path=prim_path, name=geom_name, collision=True)

    if isinstance(scale, tuple) or isinstance(scale, list):
        geom.set_local_scale(scale)
    elif isinstance(scale, float):
        geom.set_local_scale([scale, scale, scale])

    geom.set_world_pose(position=ini_pos, orientation=orientation)
    geom.set_default_state(position=ini_pos, orientation=orientation)

    if collision_type is not None:
        geom.set_collision_approximation(collision_type)
    if mass_value > 0.0:
        massAPI = UsdPhysics.MassAPI.Apply(get_current_stage().GetPrimAtPath(prim_path))
        massAPI.CreateMassAttr().Set(mass_value)

    if static_friction is not None and dynamic_friction is not None:
        add_friction_properties(
            prim_path=prim_path,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        )

    return geom



def add_cloth_particle_physics(
        stage,
        prim_path: str,
        physics_config: dict
    ):
    # configure and create particle system
    # so the simulation determines the other offsets from the particle contact offset
    particle_system_path = Sdf.Path("/PhysicsMaterials/" + prim_path.replace("/", "_").lower() + "/ParticleSystem")
    particleUtils.add_physx_particle_system(
        stage=stage,
        particle_system_path=particle_system_path,
        contact_offset=physics_config["contact_offset"],
        rest_offset=physics_config["rest_offset"],
        particle_contact_offset=physics_config["particle_contact_offset"],
        solid_rest_offset=physics_config["solid_rest_offset"],
        fluid_rest_offset=physics_config["fluid_rest_offset"],
        solver_position_iterations=physics_config["solver_position_iterations"],
    )
    
    # create material and assign it to the system:
    particle_material_path = Sdf.Path("/PhysicsMaterials/" + prim_path.replace("/", "_").lower() + "/ParticleMaterial")
    # add some drag and lift to get aerodynamic effects
    particleUtils.add_pbd_particle_material(
        stage, 
        particle_material_path, 
        drag=physics_config["drag"], 
        lift=physics_config["lift"],
        friction=physics_config["friction"],
        adhesion=physics_config["adhesion"],
        particle_adhesion_scale=physics_config["particle_adhesion_scale"],
        adhesion_offset_scale=physics_config["adhesion_offset_scale"],
        particle_friction_scale=physics_config["particle_friction_scale"],
        vorticity_confinement=physics_config["vorticity_confinement"],
        cfl_coefficient=physics_config["cfl_coefficient"],
        gravity_scale=physics_config["gravity_scale"]
    )
    physicsUtils.add_physics_material_to_prim(
        stage, stage.GetPrimAtPath(particle_system_path), particle_material_path
    )

    object_name = prim_path.split("/")[-1]
    env_name = prim_path.split("/")[-2]

    # Iterate the stage and populate lists with mesh paths.
    object_mesh_paths = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if UsdGeom.Mesh(prim) and (object_name in prim.GetPath().pathString) and (env_name in prim.GetPath().pathString):
            object_mesh_paths.append(prim.GetPath().pathString)
            print(f"[add_cloth_particle_physics] prim.GetPath() : {prim.GetPath().pathString}")

    # configure as bag
    for object_mesh_path in object_mesh_paths:
        particleUtils.add_physx_particle_cloth(
            stage=stage,
            path=object_mesh_path,
            dynamic_mesh_path=None,
            particle_system_path=particle_system_path,
            spring_stretch_stiffness=physics_config["spring_stretch_stiffness"],
            spring_bend_stiffness=physics_config["spring_bend_stiffness"],
            spring_shear_stiffness=physics_config["spring_shear_stiffness"],
            spring_damping=physics_config["spring_damping"],
            self_collision=True,
            self_collision_filter=True,
        )
    return


def add_random_object_physics(
        prim_path: str,
        is_rigid_body: bool,
        is_collision: bool,
        is_kinematic: bool,
        collision_type: str,
        mass_modify: bool,
        min_mass_value: float,
        max_mass_value: float,
        friction_modify: bool,
        min_static_friction: float,
        max_static_friction: float,
        min_dynamic_friction: float,
        max_dynamic_friction: float,
    ):

    if mass_modify:
        mass_value = np.random.uniform(min_mass_value, max_mass_value)
    else:
        mass_value = -1.0

    geom, rigid = add_physical_properties(
        prim_path=prim_path,
        rigid_body=is_rigid_body,
        collision=is_collision,
        kinematic_enabled=is_kinematic,
        collision_approximation=collision_type,
        mass_value=mass_value,
    )
    
    if friction_modify:
        static_friction = np.random.uniform(min_static_friction, max_static_friction)
        dynamic_friction = np.random.uniform(min_dynamic_friction, max_dynamic_friction)
        
        add_friction_properties(
            prim_path=prim_path,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        )
    
    return geom, rigid


def add_physical_properties(
        prim_path: str,
        collision: bool = True,
        rigid_body: bool = True,
        collision_approximation: str = "none", # convexDecomposition, convexHull, boundingCube
        kinematic_enabled: bool = False,
        mass_value: float = -1.0,
    ):
    if rigid_body:
        rigid = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(prim_path))
        if kinematic_enabled:
            rigid.CreateKinematicEnabledAttr(True)
        if mass_value > 0.0:
            object_mass = UsdPhysics.MassAPI.Apply(get_current_stage().GetPrimAtPath(prim_path))
            object_mass.CreateMassAttr().Set(mass_value)
    else:
        rigid = None
    
    if collision:
        # TODO: check for other examples
        # This codes are changed during json battery env ex
        # geom_name = prim_path.split("/")[-1]
        geom_name = prim_path.split("/")[-2] + "_" + prim_path.split("/")[-1]
        geom = GeometryPrim(prim_path=prim_path, name=geom_name, collision=True)
        geom.set_collision_approximation(collision_approximation)
    else:
        geom = None

    return geom, rigid


def add_friction_properties(
        prim_path: str,
        static_friction: float,
        dynamic_friction: float,
    ):
    # modified_prim_path = prim_path.replace("/", "_").lower()
    geom_name = prim_path.split("/")[-2] + "_" + prim_path.split("/")[-1]

    friction_material = PhysicsMaterial(
        prim_path=f"/PhysicsMaterials/{geom_name}_material",
        name=f"{geom_name}_material",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
    )
    geometry_prim = GeometryPrim(prim_path=prim_path)
    geometry_prim.apply_physics_material(friction_material)


def add_gripper_friction(
        index: int, 
        world, 
        robot_config,
        robot_name = None,
    ):

    if robot_name is None:
        robot_prim_path=f"Env_{index}/robot"
        modified_prim_path = robot_prim_path.replace("/", "_").lower()
    else:
        robot_prim_path=f"Env_{index}/{robot_name}"
        modified_prim_path = robot_prim_path.replace("/", "_").lower()

    min_finger_static_friction = robot_config["min_finger_static_friction"]
    max_finger_static_friction = robot_config["max_finger_static_friction"]
    min_finger_dynamic_friction = robot_config["min_finger_dynamic_friction"]
    max_finger_dynamic_friction = robot_config["max_finger_dynamic_friction"]

    random_finger_static_friction = random.uniform(min_finger_static_friction, max_finger_static_friction)
    random_finger_dynamic_friction = random.uniform(min_finger_dynamic_friction, max_finger_dynamic_friction)

    robot_finger_physics_material = PhysicsMaterial(
        prim_path=f"/PhysicsMaterials/{modified_prim_path}_gripper_material",
        name=f"{modified_prim_path}_gripper_material",
        static_friction=random_finger_static_friction,
        dynamic_friction=random_finger_dynamic_friction,
    )

    if "finger_names" in robot_config:
        for finger_name in robot_config["finger_names"]:
            finger_prim_path = f"/World/{robot_prim_path}/{finger_name}"
            robot_finger = world.stage.GetPrimAtPath(finger_prim_path)
            x = UsdShade.MaterialBindingAPI.Apply(robot_finger)
            x.Bind(
                robot_finger_physics_material.material,
                bindingStrength="weakerThanDescendants",
                materialPurpose="physics",
            )
    return


#### Material Functions ####

def get_textures(dir_path, png_type=".png"):
    textures = []
    dir_path += "/"
    for file in os.listdir(dir_path):
        if file.endswith(png_type):
            textures.append(dir_path + file)
    return textures


def create_defects(
        index: int,
        target_prim: str,
        texture_directory: str,
        abn_type: str = "scratches",
        semantic: str = "scratch_mesh",
        num_abn: int = 3,
        min_rotation: tuple = (-180, -180, -180),
        max_rotation: tuple = (180, 180, 180),
        min_scale: tuple = (0.1, 0.1, 0.1),
        max_scale: tuple = (0.1, 0.1, 0.1),
    ):

    texture_folder_directory = texture_directory + "/" + abn_type

    diffuse_textures = get_textures(texture_folder_directory, "_D.png")
    normal_textures = get_textures(texture_folder_directory, "_N.png")
    roughness_textures = get_textures(texture_folder_directory, "_R.png")
    num_scratch_textures = len(diffuse_textures)

    target_prim_rep = rep.get.prims(
        path_pattern=target_prim
    )

    cube_sem = [('class', f'{semantic}_{index}')]
    proj_sem = [('class', f'{semantic}_projectmat')]

    for i in range(num_abn):
        cube = rep.create.cube(
            visible=False,
            semantics=cube_sem, 
            position=0, 
            scale=0.0001,
            rotation=(0, 0, 90)
        )
        with target_prim_rep:
            rep.create.projection_material(
                cube,
                proj_sem,
                offset_scale=0.0001
            )

    defects = rep.get.prims(semantics=cube_sem)
    plane = rep.get.prim_at_path(target_prim)
    
    with defects:
        rep.randomizer.scatter_2d(plane)
        rep.modify.pose(
            rotation=rep.distribution.uniform(
                min_rotation, 
                max_rotation
            ),
            scale=rep.distribution.uniform(
                min_scale,
                max_scale
            )
        )

    projections = rep.get.prims(semantics=[('class', f'{semantic}_projectmat')])
    random_number = random.choice(
        list(range(num_scratch_textures))
    )

    with projections:
        rep.modify.projection_material(
            diffuse=diffuse_textures[random_number],
            normal=normal_textures[random_number],
            roughness=roughness_textures[random_number]
        )
    return


def create_random_omnipbr_material(
        prim_path=None,
        min_diffuse: list = [0.0, 0.0, 0.0], 
        max_diffuse: list = [1.0, 1.0, 1.0],
        min_roughness: float = 0.0,
        max_roughness: float = 1.0,
        min_metallic: float = 0.0,
        max_metallic: float = 1.0,
        min_specular: float = 0.0, 
        max_specular: float = 1.0,
        min_emissive_color: list = [0.0, 0.0, 0.0], 
        max_emissive_color: list = [1.0, 1.0, 1.0],
        min_emissive_intensity: float = 0.0,
        max_emissive_intensity: float = 1000.0,
    ):
    rep.create.material_omnipbr(
        diffuse=rep.distribution.uniform(tuple(min_diffuse), tuple(max_diffuse)),
        roughness=rep.distribution.uniform(min_roughness, max_roughness),
        metallic=rep.distribution.choice([min_metallic, max_metallic]),
        specular=rep.distribution.uniform(min_specular, max_specular),
        emissive_color=rep.distribution.uniform(tuple(min_emissive_color), tuple(max_emissive_color)),
        emissive_intensity=rep.distribution.uniform(min_emissive_intensity, max_emissive_intensity),
        count=1,
    )
    if prim_path is not None:
        # Move material to proper location
        new_prim_path = f"/Looks/PBRMaterials/" + prim_path.replace("/", "_").lower()
        omni.kit.commands.execute(
            "MovePrim", 
            path_from="/Replicator/Looks/OmniPBR", 
            path_to=new_prim_path,
        )
    return


def create_random_mdl_materaial(
        prim_path=None,
        mtl_name=None,
        category: str = "Masonry",
    ):

    if category == "Masonry":
        options = [
            "Adobe_Brick", "Brick_Pavers", "Brick_Wall_Brown", "Brick_Wall_Red",
            "Concrete_Block", "Concrete_Formed", "Concrete_Polished", "Concrete_Rough",
            "Concrete_Smooth", "Stucco"
        ]
    elif category == "Glass":
        options = [
            "Blue_Glass", "Clear_Glass", "Dull_Glass", "Frosted_Glass", "Glazed_Glass",
            "Green_Glass", "Mirror", "Red_Glass", "Tinted_Glass", "Tinted_Glass_R02",
            "Tinted_Glass_R25", "Tinted_Glass_R50", "Tinted_Glass_R75", "Tinted_Glass_R85",
            "Tinted_Glass_R98"
        ]
    elif category == "Metals":
        options = [
            "Aluminum_Anodized", "Aluminum_Anodized_Black", "Aluminum_Anodized_Blue",
            "Aluminum_Anodized_Charcoal", "Aluminum_Anodized_Red", "Aluminum_Cast",
            "Aluminum_Polished", "Brass", "Bronze", "Brushed_Antique_Copper",
            "Cast_Metal_Silver_Vein", "Chrome", "Copper", "CorrugatedMetal", "Gold",
            "Iron", "Metal_Door", "Metal_Seamed_Roof", "RustedMetal", "Silver",
            "Steel_Blued", "Steel_Carbon", "Steel_Cast", "Steel_Stainless"
        ]
    elif category == "Plastics":
        options = [
            "Plastic", "Plastic_ABS", "Plastic_Acrylic", "Plastic_Clear", "Plastic", 
            "Rubber_Smooth", "Rubber_Textured", "Veneer_OU_Walnut", "Veneer_UX_Walnut_Cherry", 
            "Veneer_Z5_Maple", "Vinyl"
        ]
    elif category == "Stone":
        options = [
            "Adobe_Octagon_Dots", "Ceramic_Smooth_Fired", "Ceramic_Tile_12", "Ceramic_Tile_18",
            "Ceramic_Tile_6", "Fieldstone", "Granite_Dark", "Granite_Light", "Gravel",
            "Gravel_River_Rock", "Marble", "Marble_Smooth", "Marble_Tile_12", "Marble_Tile_18",
            "Pea_Gravel", "Porcelain_Smooth", "Porcelain_Tile_4", "Porcelain_Tile_4_Linen",
            "Porcelain_Tile_6", "Porcelain_Tile_6_Linen", "Retaining_Block", "Slate", "Stone_Wall",
            "Terracotta", "Terrazzo"
        ]
    elif category == "Wood":
        options = [
            "Ash", "Ash_Planks", "Bamboo", "Bamboo_Planks", "Beadboard", "Birch", "Birch_Planks",
            "Cherry", "Cherry_Planks", "Cork", "Mahogany", "Mahogany_Planks", "Oak", "Oak_Planks",
            "Parquet_Floor", "Plywood", "Timber", "Timber_Cladding", "Walnut", "Walnut_Planks"
        ]

    if mtl_name is None:
        mtl_name = random.choice(options)
        mtl_url = f"omniverse://localhost/NVIDIA/Materials/Base/{category}/{mtl_name}.mdl"
    else:
        mtl_url = f"omniverse://localhost/NVIDIA/Materials/Base/{category}/{mtl_name}.mdl"

    mtl_path = f"/Looks/MDLMaterials/{category}/{mtl_name}"
    success, result = omni.kit.commands.execute(
        "CreateMdlMaterialPrimCommand",
        mtl_url=mtl_url,
        mtl_name=mtl_name,
        mtl_path=mtl_path
    )

    if prim_path is not None:
        # Move material to proper location
        new_prim_path = f"/Looks/MDLMaterials/" + prim_path.replace("/", "_").lower()
        omni.kit.commands.execute(
            "MovePrim", 
            path_from=mtl_path, 
            path_to=new_prim_path,
        )
    return


def create_random_omniglass_material(
        stage,
        prim_path: str,
        min_color: list = [0.0, 0.0, 0.0],
        max_color: list = [1.0, 1.0, 1.0],
        min_depth: float = 0.0, 
        max_depth: float = 1.0,
        min_ior: float = 1.0,
        max_ior: float = 2.0,
    ):
        mtl_created_list = []
        # Create a new material using OmniGlass.mdl
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniGlass.mdl",
            mtl_name="OmniGlass",
            mtl_created_list=mtl_created_list,
        )
        material_path = "/Looks/OmniGlassMaterials/" + prim_path.replace("/", "_").lower()
        omni.kit.commands.execute(
            "MovePrim", 
            path_from="/Looks/OmniGlass", 
            path_to=material_path,
        )
        mtl_prim = stage.GetPrimAtPath(material_path)

        color_r, color_g, color_b = random.uniform(min_color[0], max_color[0]), random.uniform(min_color[1], max_color[1]), random.uniform(min_color[2], max_color[2])
        glass_depth = random.uniform(min_depth, max_depth)
        glass_ior = random.uniform(min_ior, max_ior)

        # Set material inputs, these can be determined by looking at the .mdl file
        # or by selecting the Shader attached to the Material in the stage window and looking at the details panel
        omni.usd.create_material_input(mtl_prim, "depth", glass_depth, Sdf.ValueTypeNames.Float)
        omni.usd.create_material_input(mtl_prim, "glass_color", Gf.Vec3f(color_r, color_g, color_b), Sdf.ValueTypeNames.Color3f)
        omni.usd.create_material_input(mtl_prim, "glass_ior", glass_ior, Sdf.ValueTypeNames.Float)


def attach_material_to_prim(
        prim_path: str,
        material_path: str,
    ):
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    rep_material = UsdShade.Material(stage.GetPrimAtPath(material_path))

    UsdShade.MaterialBindingAPI(prim).Bind(
        rep_material, 
        UsdShade.Tokens.strongerThanDescendants
    )


def add_material_to_prim(
        prim_path: str,
        material,
    ):
    prim = get_current_stage().GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI(prim).Bind(
        material, 
        UsdShade.Tokens.strongerThanDescendants
    )


#### Legacy Functions ####
## Will be deleted soon ##

def create_mdl_material(
        stage,
        mtl_url: str,
        mtl_name: str,
    ):
    success, result = omni.kit.commands.execute(
        "CreateMdlMaterialPrimCommand",
        mtl_url=mtl_url,
        mtl_name=mtl_name,
        mtl_path=f"/World/Looks/{mtl_name}"
    )
    material = UsdShade.Material(stage.GetPrimAtPath(f"/World/Looks/{mtl_name}"))
    return material


def create_aluminum_polished_material(stage):
    # Aluminum_Polished
    material = create_mdl_material(
        stage,
        mtl_url="omniverse://localhost/NVIDIA/Materials/Base/Metals/Aluminum_Polished.mdl",
        mtl_name="Aluminum_Polished",
    )
    return material


def create_copper_material(stage):
    # Copper
    material = create_mdl_material(
        stage,
        mtl_url="omniverse://localhost/NVIDIA/Materials/Base/Metals/Copper.mdl",
        mtl_name="Copper",
    )
    return material 


def create_iron_material(stage):
    # Iron
    material = create_mdl_material(
        stage,
        mtl_url="omniverse://localhost/NVIDIA/Materials/Base/Metals/Iron.mdl",
        mtl_name="Iron",
    )
    return material 


def create_glass_material(
        stage,
        depth: float = 0.5,
        glass_color: tuple = (1, 1, 1),
        glass_ior: float = 2.0,
    ):
    mtl_created_list = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniGlass.mdl",
        mtl_name="OmniGlass",
        mtl_created_list=mtl_created_list,
    )
    mtl_prim = stage.GetPrimAtPath(mtl_created_list[0])

    color_r, color_g, color_b = glass_color
    omni.usd.create_material_input(mtl_prim, "depth", depth, Sdf.ValueTypeNames.Float)
    omni.usd.create_material_input(mtl_prim, "glass_color", Gf.Vec3f(color_r, color_g, color_b), Sdf.ValueTypeNames.Color3f)
    omni.usd.create_material_input(mtl_prim, "glass_ior", glass_ior, Sdf.ValueTypeNames.Float)
    return UsdShade.Material(mtl_prim)


def add_sphere_light(
        stage,
        prim_path,
        intensity,
        translate,
        scale,
    ):

    ligth_src = UsdLux.SphereLight.Define(stage, Sdf.Path(prim_path))
    ligth_src.CreateIntensityAttr(intensity)
    tx, ty, tz = tuple(translate)
    ligth_src.CreateRadiusAttr(scale)
    ligth_src.AddRotateXYZOp().Set((0, 0, 0))
    ligth_src.AddTranslateOp().Set(Gf.Vec3f(tx, ty, tz))

    return ligth_src
