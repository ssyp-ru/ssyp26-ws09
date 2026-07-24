extends Node3D

@export var player_camera: Camera3D

@onready var sub_viewport: SubViewport = $MeshInstance3D/SubViewport
@onready var mirror_camera: Camera3D = $MeshInstance3D/SubViewport/Camera3D

func _ready() -> void:
	pass
func _process(_delta: float) -> void:
	if not player_camera: 
		return
	
	var mirror_trans = global_transform  # Расположение центра зеркала
	var local_cam_pos = mirror_trans.affine_inverse() * player_camera.global_position  # произведение инвертированного расположения центра зеркала и расположения привязанной камеры
	var local_reflect_pos = Vector3(local_cam_pos.x, local_cam_pos.y, local_cam_pos.z)
	var local_cam_forward = mirror_trans.basis.inverse() * -player_camera.global_transform.basis.z
	var local_reflect_forward = Vector3(local_cam_forward.x, local_cam_forward.y, local_cam_forward.z)
	var global_reflect_forward = mirror_trans.basis * local_reflect_forward
	
	mirror_camera.global_position = mirror_trans * local_reflect_pos
	mirror_camera.look_at(mirror_camera.global_position + global_reflect_forward, mirror_trans.basis.y)
