extends GridMap



func _ready() -> void:
	var n = 30
	var height = 3
	
	for h in range(1, height + 1):
		for i in range(n + 1):
			set_cell_item(Vector3i(-i, h, 0), randi_range(0, 2))
			
			if i > 0:
				set_cell_item(Vector3i(0, h, -i), randi_range(0, 2))
				
			set_cell_item(Vector3i(-i, h, -n), randi_range(0, 2))
			
			if i < n:
				set_cell_item(Vector3i(-n, h, -i), randi_range(0, 2))


func _process(delta: float) -> void:
	pass
