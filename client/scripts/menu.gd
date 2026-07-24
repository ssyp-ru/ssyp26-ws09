extends Control

@onready var list_of_mazes: Control = $list_of_mazes
@onready var control: Control = $"."

const Builds_Path = "godot_builds"

func launch_compiled_game(base_path: String, game_name: String) -> int:
	var executable_path: String = ""
	var current_os: String = OS.get_name()

	var full_base_path: String = base_path.path_join(game_name)

	match current_os:
		"Windows":
			executable_path = full_base_path + ".exe"
		"Linux", "FreeBSD":
			executable_path = full_base_path + ".x86_64" # или .x86, или без расширения
		"macOS":
			executable_path = full_base_path + ".app/Contents/MacOS/" + game_name
		_:
			push_error("Критическая ошибка: Данная ОС не поддерживается для запуска!")
			return -1

	if not FileAccess.file_exists(executable_path) and current_os != "macOS":
		push_error("Файл не найден по пути: " + executable_path)
		return -1

	print("Запуск скомпилированной игры: ", executable_path)

	var arguments: PackedStringArray = [] 
	var pid: int = OS.execute(executable_path, arguments, [], false, false)
	
	if pid > 0:
		print("Игра успешно запущена! Процесс ID (PID): ", pid)
	else:
		push_error("Не удалось запустить процесс. Проверьте права доступа (chmod +x на Linux).")

	return pid

func _on_game1_pressed() -> void:
	launch_game(Builds_Path)

func launch_game(game_dir : String) -> void:
	var abs_game_path = ProjectSettings.globalize_path("res://").path_join("../").path_join(game_dir).simplify_path()
	var output = []
	var err = launch_compiled_game(abs_game_path, "demo")
	
	if err != OK:
		print('Не удалось запустить игру.', err)


func _on_button_pressed() -> void:
	_on_game1_pressed()

func _on_button_3_pressed() -> void:
	get_tree().quit()
