import asyncio
import json

import websockets
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8080

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000


DB_PATH = Path("DB")

external_ws_connection = None
client = None

current_maze = None


async def handler(websocket):
    global client, current_maze

    client = websocket
    print("Клиент успешно подключился")

    try:
        async for message in websocket:
            data = json.loads(message)
            COMMAND = data.get("command")

            if COMMAND == "get_list_of_mazes":
                res = []

                if DB_PATH.exists():
                    for pd in sorted(DB_PATH.iterdir()):
                        if pd.is_dir():
                            for p in sorted(pd.iterdir()):
                                if p.is_file() and p.suffix == ".json":
                                    g_num = ''.join(filter(str.isdigit, pd.name))
                                    c_num = ''.join(filter(str.isdigit, p.name))
                                    res.append(f"gen_{g_num}/chr_{c_num}")

                await client.send(json.dumps({"type": "list_of_mazes", "content": res}))

            elif COMMAND == "get_maze":
                maze_id = data.get("maze_id")

                g, c = maze_id.split('/')
                g, c = int(g[4:]), int(c[4:])

                path_to_maze = DB_PATH.joinpath(f"generation_{g}").joinpath(f"chromosome_{c}.json")

                if path_to_maze.is_file():
                    with open(path_to_maze, 'r') as file:
                        maze = json.load(file)

                    current_maze = maze

                    await client.send(json.dumps({"type": "maze"} | maze))
                else:
                    await client.send(json.dumps({"error": f"maze {maze_id} not found on disk"}))

            elif COMMAND == "get_ai_moves":
                if current_maze is None:
                    await client.send(json.dumps({"error": "undefined maze"}))
                    continue

                if external_ws_connection is not None:
                    await external_ws_connection.send(json.dumps({"command": "set_maze", "maze": current_maze}))
                    await external_ws_connection.send(json.dumps({"command": "get_actions", "actions_cnt": 4096}))
                else:
                    await client.send(json.dumps({"error": "AI Backend server is disconnected"}))

    except websockets.exceptions.ConnectionClosedOK:
        print("Соединение закрыто клиентом штатно")
    except websockets.exceptions.ConnectionClosedError:
        print("Соединение с клиентом разорвано аварийно")
        print("Клиент бездарь")
    except Exception as e:
        print(f"Непредвиденная ошибка в обработчике клиента: {e}")


async def listen_external_server():
    global external_ws_connection, client

    while True:
        try:
            print(f"Пытаемся подключиться к внешнему серверу ИИ: ws://{SERVER_HOST}:{SERVER_PORT}...")
            async with websockets.connect(f"ws://{SERVER_HOST}:{SERVER_PORT}") as external_ws:
                external_ws_connection = external_ws
                print("Успешно подключились к серверу с ИИ-агентом!")

                async for message in external_ws:
                    data = json.loads(message)
                    TYPE = data.get("type")

                    if client is not None:
                        if TYPE == "maze":
                            await client.send(json.dumps(data))

                        elif TYPE == "actions":
                            moves_data = data.get("moves", [])
                            moves = [e.get("action") for e in moves_data if "action" in e]
                            await client.send(json.dumps({"type": "actions", "content": moves}))

                        elif TYPE == "maze_list":
                            await client.send(json.dumps(data))

        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError):
            print("Соединение с внешним сервером ИИ потеряно. Повтор через 5 секунд...")
            external_ws_connection = None
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Ошибка во время удержания связи с ИИ-сервером: {e}")
            print("Вы бездарь")
            await asyncio.sleep(5)


async def main():
    asyncio.create_task(listen_external_server())

    try:
        server = await websockets.serve(handler, HOST, PORT)
        print(f"🚀 [ШЛЮЗ] Промежуточный сервер успешно запущен на ws://{HOST}:{PORT}")

        await server.wait_closed()
        
    except OSError as e:
        if e.errno == 98:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Порт {PORT} уже намертво занят другим процессом!")
            print("👉 Запустите в терминале: sudo lsof -i :8080, а затем убейте процесс через kill -9")
        else:
            print(f"❌ Системная ошибка при запуске сервера: {e}")
    except Exception as e:
        print(f"❌ Непредвиденный сбой инициализации: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nШлюз-сервер успешно остановлен пользователем (Ctrl+C).")


