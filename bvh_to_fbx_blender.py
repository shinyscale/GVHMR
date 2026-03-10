"""Blender Python script: import BVH, export FBX with DCC-compatible naming.

Run via:
  blender --background --python bvh_to_fbx_blender.py -- input.bvh output.fbx [--fps 30] [--naming mixamo|ue5] [--target blender|maya|cascadeur|ue5]

This script is executed inside Blender's Python environment.
Bones are renamed to a standard convention so Cascadeur and other DCC tools auto-detect the rig.
The --target flag selects axis/scale conventions for the destination DCC tool.
"""

import sys
import os

# Parse args after "--"
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    print("Usage: blender --background --python bvh_to_fbx_blender.py -- input.bvh output.fbx [--fps 30] [--naming mixamo|ue5] [--target blender|maya|cascadeur|ue5]")
    sys.exit(1)

bvh_path = argv[0]
fbx_path = argv[1]

fps = 30.0
if "--fps" in argv:
    fps_idx = argv.index("--fps")
    if fps_idx + 1 < len(argv):
        fps = float(argv[fps_idx + 1])

naming = "mixamo"
if "--naming" in argv:
    naming_idx = argv.index("--naming")
    if naming_idx + 1 < len(argv):
        naming = argv[naming_idx + 1].lower()

target = "cascadeur"
if "--target" in argv:
    target_idx = argv.index("--target")
    if target_idx + 1 < len(argv):
        target = argv[target_idx + 1].lower()

# ── DCC target presets — axis/scale conventions ──

TARGET_PRESETS = {
    "blender": {
        "axis_forward": '-Z', "axis_up": 'Y',
        "primary_bone_axis": 'Y', "secondary_bone_axis": 'X',
        "apply_scale_options": 'FBX_SCALE_NONE', "global_scale": 1.0,
    },
    "maya": {
        "axis_forward": '-Z', "axis_up": 'Y',
        "primary_bone_axis": 'Y', "secondary_bone_axis": 'X',
        "apply_scale_options": 'FBX_SCALE_NONE', "global_scale": 1.0,
    },
    "cascadeur": {
        "axis_forward": '-Z', "axis_up": 'Y',
        "primary_bone_axis": 'Y', "secondary_bone_axis": 'X',
        "apply_scale_options": 'FBX_SCALE_UNITS', "global_scale": 1.0,
    },
    "ue5": {
        "axis_forward": 'X', "axis_up": 'Z',
        "primary_bone_axis": 'Y', "secondary_bone_axis": 'X',
        "apply_scale_options": 'FBX_SCALE_UNITS', "global_scale": 1.0,
    },
}

# ── Bone naming conventions ──

SMPLX_TO_MIXAMO = {
    # Body
    "Pelvis": "mixamorig:Hips",
    "L_Hip": "mixamorig:LeftUpLeg",
    "R_Hip": "mixamorig:RightUpLeg",
    "Spine1": "mixamorig:Spine",
    "L_Knee": "mixamorig:LeftLeg",
    "R_Knee": "mixamorig:RightLeg",
    "Spine2": "mixamorig:Spine1",
    "L_Ankle": "mixamorig:LeftFoot",
    "R_Ankle": "mixamorig:RightFoot",
    "Spine3": "mixamorig:Spine2",
    "L_Foot": "mixamorig:LeftToeBase",
    "R_Foot": "mixamorig:RightToeBase",
    "Neck": "mixamorig:Neck",
    "L_Collar": "mixamorig:LeftShoulder",
    "R_Collar": "mixamorig:RightShoulder",
    "Head": "mixamorig:Head",
    "L_Shoulder": "mixamorig:LeftArm",
    "R_Shoulder": "mixamorig:RightArm",
    "L_Elbow": "mixamorig:LeftForeArm",
    "R_Elbow": "mixamorig:RightForeArm",
    "L_Wrist": "mixamorig:LeftHand",
    "R_Wrist": "mixamorig:RightHand",
    # Left hand
    "L_Index1": "mixamorig:LeftHandIndex1",
    "L_Index2": "mixamorig:LeftHandIndex2",
    "L_Index3": "mixamorig:LeftHandIndex3",
    "L_Middle1": "mixamorig:LeftHandMiddle1",
    "L_Middle2": "mixamorig:LeftHandMiddle2",
    "L_Middle3": "mixamorig:LeftHandMiddle3",
    "L_Pinky1": "mixamorig:LeftHandPinky1",
    "L_Pinky2": "mixamorig:LeftHandPinky2",
    "L_Pinky3": "mixamorig:LeftHandPinky3",
    "L_Ring1": "mixamorig:LeftHandRing1",
    "L_Ring2": "mixamorig:LeftHandRing2",
    "L_Ring3": "mixamorig:LeftHandRing3",
    "L_Thumb1": "mixamorig:LeftHandThumb1",
    "L_Thumb2": "mixamorig:LeftHandThumb2",
    "L_Thumb3": "mixamorig:LeftHandThumb3",
    # Right hand
    "R_Index1": "mixamorig:RightHandIndex1",
    "R_Index2": "mixamorig:RightHandIndex2",
    "R_Index3": "mixamorig:RightHandIndex3",
    "R_Middle1": "mixamorig:RightHandMiddle1",
    "R_Middle2": "mixamorig:RightHandMiddle2",
    "R_Middle3": "mixamorig:RightHandMiddle3",
    "R_Pinky1": "mixamorig:RightHandPinky1",
    "R_Pinky2": "mixamorig:RightHandPinky2",
    "R_Pinky3": "mixamorig:RightHandPinky3",
    "R_Ring1": "mixamorig:RightHandRing1",
    "R_Ring2": "mixamorig:RightHandRing2",
    "R_Ring3": "mixamorig:RightHandRing3",
    "R_Thumb1": "mixamorig:RightHandThumb1",
    "R_Thumb2": "mixamorig:RightHandThumb2",
    "R_Thumb3": "mixamorig:RightHandThumb3",
}

SMPLX_TO_UE5 = {
    # Root locomotion bone
    "Root": "root",
    # Body — UE5 Mannequin (Manny/Quinn) naming
    "Pelvis": "pelvis",
    "L_Hip": "thigh_l",
    "R_Hip": "thigh_r",
    "Spine1": "spine_01",
    "L_Knee": "calf_l",
    "R_Knee": "calf_r",
    "Spine2": "spine_02",
    "L_Ankle": "foot_l",
    "R_Ankle": "foot_r",
    "Spine3": "spine_03",
    "L_Foot": "ball_l",
    "R_Foot": "ball_r",
    "Neck": "neck_01",
    "L_Collar": "clavicle_l",
    "R_Collar": "clavicle_r",
    "Head": "head",
    "L_Shoulder": "upperarm_l",
    "R_Shoulder": "upperarm_r",
    "L_Elbow": "lowerarm_l",
    "R_Elbow": "lowerarm_r",
    "L_Wrist": "hand_l",
    "R_Wrist": "hand_r",
    # Left hand
    "L_Index1": "index_01_l",
    "L_Index2": "index_02_l",
    "L_Index3": "index_03_l",
    "L_Middle1": "middle_01_l",
    "L_Middle2": "middle_02_l",
    "L_Middle3": "middle_03_l",
    "L_Pinky1": "pinky_01_l",
    "L_Pinky2": "pinky_02_l",
    "L_Pinky3": "pinky_03_l",
    "L_Ring1": "ring_01_l",
    "L_Ring2": "ring_02_l",
    "L_Ring3": "ring_03_l",
    "L_Thumb1": "thumb_01_l",
    "L_Thumb2": "thumb_02_l",
    "L_Thumb3": "thumb_03_l",
    # Right hand
    "R_Index1": "index_01_r",
    "R_Index2": "index_02_r",
    "R_Index3": "index_03_r",
    "R_Middle1": "middle_01_r",
    "R_Middle2": "middle_02_r",
    "R_Middle3": "middle_03_r",
    "R_Pinky1": "pinky_01_r",
    "R_Pinky2": "pinky_02_r",
    "R_Pinky3": "pinky_03_r",
    "R_Ring1": "ring_01_r",
    "R_Ring2": "ring_02_r",
    "R_Ring3": "ring_03_r",
    "R_Thumb1": "thumb_01_r",
    "R_Thumb2": "thumb_02_r",
    "R_Thumb3": "thumb_03_r",
}

NAMING_MAPS = {
    "mixamo": SMPLX_TO_MIXAMO,
    "ue5": SMPLX_TO_UE5,
}

# ── Blender operations ──

import bpy
import mathutils

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Set scene FPS
bpy.context.scene.render.fps = int(fps)
bpy.context.scene.render.fps_base = 1.0

# Set Blender scene units to centimeters to match Cascadeur
# Blender's default is meters (unit_scale=1.0).
# Setting unit_scale=0.01 means 1 Blender unit = 0.01m = 1cm.
bpy.context.scene.unit_settings.scale_length = 0.01

# Import BVH — keep cm scale (global_scale=1.0) since scene is now in cm
bpy.ops.import_anim.bvh(
    filepath=bvh_path,
    filter_glob="*.bvh",
    global_scale=1.0,
    use_fps_scale=False,
    rotate_mode='NATIVE',
)

# Find the imported armature
armature = None
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        armature = obj
        break

if armature is None:
    print("[BVH→FBX] ERROR: No armature found after BVH import")
    sys.exit(1)

# Select only the armature
bpy.ops.object.select_all(action='DESELECT')
armature.select_set(True)
bpy.context.view_layer.objects.active = armature

# Apply all transforms so the armature has scale (1,1,1)
# Cascadeur requires root scale = 1,1,1
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

print(f"[BVH→FBX] Armature scale after apply: {list(armature.scale)}")

# Rename armature object
armature.name = "Armature"

# Rename bones to target convention (edit mode so animation fcurves update automatically)
rename_map = NAMING_MAPS.get(naming, {})
if rename_map:
    bpy.ops.object.mode_set(mode='EDIT')
    renamed = 0
    for bone in armature.data.edit_bones:
        new_name = rename_map.get(bone.name)
        if new_name:
            bone.name = new_name
            renamed += 1
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[BVH→FBX] Renamed {renamed}/{len(armature.data.bones)} bones to {naming} convention")
else:
    print(f"[BVH→FBX] Unknown naming '{naming}', keeping SMPL-X names")

# Set frame range from animation
if armature.animation_data and armature.animation_data.action:
    action = armature.animation_data.action
    frame_start, frame_end = action.frame_range
    bpy.context.scene.frame_start = int(frame_start)
    bpy.context.scene.frame_end = int(frame_end)

print(f"[BVH→FBX] Armature: {armature.name}")
print(f"[BVH→FBX] Bones: {len(armature.data.bones)}")
print(f"[BVH→FBX] Frames: {bpy.context.scene.frame_start}-{bpy.context.scene.frame_end}")
print(f"[BVH→FBX] FPS: {fps}")
print(f"[BVH→FBX] Naming: {naming}")

# Resolve DCC target preset
preset = TARGET_PRESETS.get(target, TARGET_PRESETS["cascadeur"])
if target not in TARGET_PRESETS:
    print(f"[BVH→FBX] WARNING: Unknown target '{target}', falling back to cascadeur")
print(f"[BVH→FBX] Target: {target} → {preset}")

# Export FBX with target-specific axis/scale settings
bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=True,
    object_types={'ARMATURE'},
    add_leaf_bones=False,
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
    bake_anim_simplify_factor=0.0,  # No simplification — keep all keyframes
    **preset,
)

print(f"[BVH→FBX] Exported: {fbx_path}")
print(f"[BVH→FBX] Size: {os.path.getsize(fbx_path) / 1024:.1f} KB")
