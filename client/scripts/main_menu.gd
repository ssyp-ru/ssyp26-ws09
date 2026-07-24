class_name Skebob
extends Control

@onready var menu: Control = $Control
@onready var list_of_maze: Control = $Control2


func _on_button_2_pressed() -> void:
	SceneManager.send_command({"command": "get_list_of_mazes"})
	
	menu.hide()
	list_of_maze.show()


func _on_to_main_menu_pressed() -> void:
	menu.visible = true
	list_of_maze.visible = false


func _on_ai_play_pressed() -> void:
	SceneManager.send_command({"command": "get_ai_moves"})
	
	SceneManager.switch_scene("maze")
