class_name Stepper
extends CharacterBody3D
@onready var player: CharacterBody3D = $"."
@onready var camera_3d: Camera3D = $Camera3D
@export var mouse_sensitivity: float = 0.003
@export var follow_smoothness: float = 5.0 # Чем выше, тем быстрее камера догоняет цель
@export var STEP_SIZE = 2.

var index_target : int = 0
# Внутренние переменные
var target_position: Vector3 = Vector3.ZERO
var yaw: float = 0.0 # Вращение влево-вправо (ось Y)
var pitch: float = 0.0 # Вращение вверх-вниз (ось X)

var queue: Array = [] 

const SPEED = 5.0

var gravity = ProjectSettings.get_setting("physics/3d/default_gravity")

const JUMP_VELOCITY = 5.85
var time: float = 0.0
@export var rot_speed = PI

var target_transform: Transform3D
var first_exec: bool = true
var move: Vector3

var play : bool = false

func _ready() -> void:
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	
	move = Vector3.ZERO
	
	target_transform = camera_3d.transform


func _input(event):
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * mouse_sensitivity)
		camera_3d.rotate_x(-event.relative.y * mouse_sensitivity)
		# Ограничиваем обзор по вертикали
		camera_3d.rotation.x = clamp(camera_3d.rotation.x, deg_to_rad(-89), deg_to_rad(89))


func _physics_process(delta: float) -> void:
	time += delta
	if not is_on_floor():
		velocity.y -= gravity * delta

	# Handle jump.
	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		velocity.y = JUMP_VELOCITY

	if Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT) and Input.MOUSE_MODE_CAPTURED:
		velocity.y += SPEED/10
	# Get the input direction and handle the movement/deceleration.
	# As good practice, you should replace UI actions with custom gameplay actions.
	var input_dir := Input.get_vector("ui_left", "ui_right", "ui_up", "ui_down")
	var direction := (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	if direction:
		velocity.x += direction.x * SPEED * delta
		velocity.z += direction.z * SPEED * delta
	else:
		velocity.x += move_toward(velocity.x, 0, SPEED)
		velocity.z += move_toward(velocity.z, 0, SPEED)

	move_and_slide()
	
	
func _process(delta: float) -> void:
	time += delta
	if time > 20:
		if Input.is_action_just_pressed("ui_right"):
			play = true
		if queue != [] and play == false:
			global_position.x += move.x * delta * 10
			global_position.z += move.z * delta * 10
			
			camera_3d.transform = camera_3d.transform.interpolate_with(target_transform, delta * rot_speed)
			
			if first_exec or (abs((queue[index_target]-global_position).x) < 0.2 and abs((queue[index_target]-global_position).z) < 0.2):
				if index_target < len(queue) - 1:
					index_target+=1
					#if index_target % 10 == 0:
					print(index_target)
					
				
					move = (queue[index_target]-global_position).normalized()
					move.y = 1.1
				
					first_exec = false
				
					target_transform = camera_3d.transform.looking_at(move)
				else:
					print('Winx!')
		
		
func add_target_pos_to_queue(target: Vector3) -> void:
	queue.append(target)
	print(target)
	
func add_move_to_queue(move: Vector3):
	if not queue.is_empty():
		add_target_pos_to_queue(queue.back() + move * STEP_SIZE)
		
	else:
		add_target_pos_to_queue(global_position + move * STEP_SIZE)
	
func add_moves_to_queue(moves: Array):
	for move in moves:
		add_move_to_queue(move)
