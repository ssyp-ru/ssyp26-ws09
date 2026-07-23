import asyncio
import json
import torch
import numpy as np
import websockets
from gym.spaces import Box, Discrete

from ai.model import RLTwoLayerCfCModel
from ai.environment import MazeEnvironment

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 6543

class AgentInference:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        obs_space = Box(low=-1.0, high=2.0, shape=(61,), dtype=np.float32)
        action_space = Discrete(4)
        num_outputs = 4
        model_config = {}

        self.model = RLTwoLayerCfCModel(obs_space, action_space, num_outputs, model_config, "inference_model")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "worker" in checkpoint and "policy_map" in checkpoint["worker"]:
            state_dict = checkpoint["worker"]["policy_map"]["default_policy"]["model"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.current_state = [s.to(self.device) for s in self.model.get_initial_state()]
        
    def reset_memory(self):
        self.current_state = [s.to(self.device) for s in self.model.get_initial_state()]

    def get_action(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            input_dict = {"obs": obs_tensor}

            prepared_state = []
            for s in self.current_state:
                if s.dim() == 1:
                    prepared_state.append(s.unsqueeze(0))
                else:
                    prepared_state.append(s)

            logits, next_state = self.model(input_dict, prepared_state, seq_lens=None)
            action = torch.argmax(logits, dim=-1).item()

            self.current_state = [s.squeeze(0) for s in next_state]
            return action


async def websocket_handler(websocket, agent: AgentInference, env: MazeEnvironment):
    print(f"Промежуточный шлюз подключился: {websocket.remote_address}")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            COMMAND = data.get("command")

            if COMMAND == "set_maze":
                maze_data = data.get("maze")
    
                env.chromosome_apply_mode = False
                env.current_maze = maze_data
                
                print("Лабиринт успешно обновлен в MazeEnvironment через WebSocket.")
                continue

            elif COMMAND == "get_actions":
                actions_cnt = data.get("actions_cnt", 4096)
 
                obs, info = env.reset()
                agent.reset_memory()
                
                moves_list = []
   
                for step_idx in range(actions_cnt):
                    act = agent.get_action(obs)
       
                    moves_list.append({"action": act})
                    
                    obs, reward, done, truncated, info = env.step(act)
                    
                    if done or truncated:
                        break

                response = {
                    "type": "actions",
                    "moves": moves_list
                }
                await websocket.send(json.dumps(response))
                print(f"Сгенерировано и отправлено траекторий шагов: {len(moves_list)}")
            
    except websockets.exceptions.ConnectionClosedOK:
        print("Шлюз отключился штатно.")
    except websockets.exceptions.ConnectionClosedError:
        print("Соединение со шлюзом разорвано аварийно.")
    except Exception as e:
        print(f"Ошибка в WebSocket сессии ИИ-сервера: {e}")


async def main():
    CHECKPOINT_PATH = "model_gen_65.pth" 

    agent = AgentInference(checkpoint_path=CHECKPOINT_PATH, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Модель успешно загружена на устройство: {agent.device}")

    env_config = {"maze_size": 128, "diamond_radius": 5, "maze_manager": None}
    env = MazeEnvironment(config=env_config)

    print(f"\n🚀 Запуск WebSocket сервера ИИ на ws://{SERVER_HOST}:{SERVER_PORT}...")

    async with websockets.serve(lambda ws: websocket_handler(ws, agent, env), SERVER_HOST, SERVER_PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

