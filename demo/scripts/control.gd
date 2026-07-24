extends Control

func _input(event) -> void:
	if event.is_action_pressed("ui_cancel"):
		visible = !visible
		get_tree().paused = visible
		
		if visible:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		else:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func _on_button_pressed() -> void:
	get_tree().quit()


func _on_button_2_pressed() -> void:
	visible = !visible
	get_tree().paused = visible
	
	if visible:
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	else:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
