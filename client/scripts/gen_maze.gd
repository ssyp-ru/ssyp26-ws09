class_name GenMaze
extends Node3D

@onready var grid_map: GridMap = $GridMap
@onready var agent: Stepper = $Agent

const HEIGHT = 3

var start: Array = [0, 0]
var finish: Array = [0, 0]

func _ready() -> void:
	SceneManager.ai_moves_recieved.connect(_on_ai_moves_recieved)


func reset() -> void:
	grid_map.clear()


func set_start(s: Array) -> void:
	var sss = grid_map.to_global(grid_map.map_to_local(Vector3(s[0], 0, s[1])))
	start = [sss.x, sss.z]
	agent.global_position = Vector3(start[0], 3, start[1])

	grid_map.set_cell_item(Vector3i(start[0], 10, start[1]), 2)
	agent.velocity = Vector3.ZERO


func set_finish(f: Array) -> void:
	var  fff = grid_map.to_global(grid_map.map_to_local(Vector3(f[0], 0, f[1])))
	finish = [fff.x, fff.z]
	
	grid_map.set_cell_item(Vector3i(finish[0], 10, finish[1]), 2)


func set_maze_mat(mat: Array):
	var size = mat.size()
	
	for k in range(size):
		for l in range(size):
			grid_map.set_cell_item(Vector3i(k, 0, l), 0)
			
			if mat[k][l] == 1:
				for h in range(HEIGHT):
					grid_map.set_cell_item(Vector3i(k, h + 1, l), 1)
	
	for k in range(size):
		for h in range(HEIGHT):
			grid_map.set_cell_item(Vector3i(0, h, k), 0)
			
			if k != 0:
				grid_map.set_cell_item(Vector3i(k, h, 0), 0)
				
				grid_map.set_cell_item(Vector3i(size - 1, h, k), 0)
				
				if k != size - 1:
					grid_map.set_cell_item(Vector3i(k, h, size - 1), 0)
			

func from_dict(data: Dictionary) -> void:
	set_start(data.get("start"))
	set_finish(data.get("finish"))
	
	set_maze_mat(data.get("mat"))

func _on_ai_moves_recieved(recieved: bool):
	from_dict(SceneManager.current_maze_data)
	
	var moves = [Vector3(0, 0, -1), Vector3(0, 0, 1), Vector3(-1, 0, 0), Vector3(1, 0, 0)]
	
	for action in SceneManager.ai_actions:
		agent.add_move_to_queue(moves[action])
		print(action)
