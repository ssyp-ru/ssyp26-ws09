extends Node

const HOST = "127.0.0.1"
const PORT = 8080
var socket = WebSocketPeer.new()
var is_connected_to_server = false

const SCENES = {
	"menu": "res://scenes/MazeMenu.tscn",
	"maze": "res://scenes/Maze.tscn"
}

var mazes_list: Array = []
var current_maze_data: Dictionary = {}
var ai_actions: Array = []
var maze_mat_size: int = 128

signal maze_data_load(data: Dictionary)
signal mazes_list_updated(updated: bool)
signal ai_moves_recieved(recieved: bool)

func _ready():
	connect_to_gateway()


func _process(_delta):
	socket.poll()
	var state = socket.get_ready_state()
	
	if state == WebSocketPeer.STATE_OPEN:
		if not is_connected_to_server:
			is_connected_to_server = true
			print("[СЕТЬ] Успешно подключено к Python-шлюзу!")

		while socket.get_available_packet_count() > 0:
			_on_data_received(socket.get_packet().get_string_from_utf8())
			
	elif state == WebSocketPeer.STATE_CLOSED:
		if is_connected_to_server:
			is_connected_to_server = false
			print("[СЕТЬ] Соединение со шлюзом разорвано.")


func connect_to_gateway():
	var url = "ws://" + HOST + ":" + str(PORT)
	print("[СЕТЬ] Пытаемся подключиться к " + url + "...")
	socket.connect_to_url(url)


func send_command(command_dict: Dictionary):
	if socket.get_ready_state() == WebSocketPeer.STATE_OPEN:
		var json_str = JSON.stringify(command_dict)
		socket.send_text(json_str)
	else:
		print("[СЕТЬ] Ошибка: Попытка отправить команду на отключенный сокет.")


func _on_data_received(json_string: String):
	var json = JSON.new()
	var error = json.parse(json_string)
	if error != OK:
		print("[СЕТЬ] Ошибка парсинга JSON: ", json.get_error_message())
		return
		
	var data = json.get_data()
	var type = data.get("type")
	
	match type:
		"list_of_mazes":
			mazes_list = data.get("content", [])
			print("[СЕТЬ] Получен список лабиринтов: ", mazes_list.size(), " шт.")
			
			mazes_list_updated.emit(true)
			
		"maze":
			current_maze_data = data
			print("[СЕТЬ] Данные лабиринта загружены. Отрисовка миниатюры...")
			
			maze_data_load.emit(current_maze_data)
			# switch_scene("maze")
			
		"actions":
			ai_actions = data.get("content", [])
			print("[СЕТЬ] Массив ходов ИИ получен! Шагов: ", ai_actions.size())

			if get_tree().current_scene.has_method("_on_ai_actions_ready"):
				get_tree().current_scene._on_ai_actions_ready(ai_actions)
				
			ai_moves_recieved.emit(true)
				
		"maze_mat_size":
			maze_mat_size = data.get("content")


func switch_scene(scene_key: String):
	if SCENES.has(scene_key):

		get_tree().call_deferred("change_scene_to_file", SCENES[scene_key])
		print("[МЕНЕДЖЕР] Сцена изменена на: ", scene_key)
	else:
		print("[МЕНЕДЖЕР] Ошибка: Сцена с ключом '", scene_key, "' не зарегистрирована.")
