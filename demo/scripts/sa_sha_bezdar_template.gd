extends Node3D

@onready var mesh_instance: MeshInstance3D = $MeshInstance3D
@onready var video_player: VideoStreamPlayer = $SubViewport/VideoStreamPlayer

@export var video_source: VideoStream

func _ready() -> void:
	# Запускаем видео, иначе текстуры не будет
	video_player.stream = video_source
	video_player.play()
	
	# Ждем один кадр, пока декодер создаст первый фрейм
	await RenderingServer.frame_post_draw
	
	# Получаем текущий материал меша (индекс 0 — первая поверхность)
	var material: StandardMaterial3D = mesh_instance.mesh.surface_get_material(0) as StandardMaterial3D
	if material:
		# Назначаем текстуру напрямую из плеера
		material.albedo_texture = video_player.get_video_texture()
