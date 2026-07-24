extends CharacterBody3D

# Настройки движения
const SPEED = 5.0
const JUMP_VELOCITY = 5.85
const MOUSE_SENSITIVITY = 0.003

# Гравитация из настроек проекта
var gravity = ProjectSettings.get_setting("physics/3d/default_gravity")

# Ссылка на камеру для обзора
@onready var camera = $Camera3D


func _ready():
	# Захватываем мышь при старте игры
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

func _unhandled_input(event):
	# Поворот камеры мышью
	if event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		rotate_y(-event.relative.x * MOUSE_SENSITIVITY)
		camera.rotate_x(-event.relative.y * MOUSE_SENSITIVITY)
		# Ограничиваем обзор по вертикали
		camera.rotation.x = clamp(camera.rotation.x, deg_to_rad(-89), deg_to_rad(89))
	
	# Освобождение мыши по нажатию Esc
#	if event.is_action_pressed("ui_cancel"):
#		
#		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
#			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
#			
#		else:
#			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

func _physics_process(delta):
	# Добавляем гравитацию, если персонаж в воздухе
	if not is_on_floor():
		velocity.y -= gravity * delta

	# Обработка прыжка
	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		velocity.y = JUMP_VELOCITY

	# Получаем вектор направления на основе стандартных UI-действий (WASD/Стрелки)
	var input_dir = Input.get_vector("ui_left", "ui_right", "ui_up", "ui_down")
	
	# Рассчитываем направление относительно взгляда персонажа
	var direction = (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	
	if direction:
		velocity.x = direction.x * SPEED
		velocity.z = direction.z * SPEED
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)
		velocity.z = move_toward(velocity.z, 0, SPEED)

	move_and_slide()
