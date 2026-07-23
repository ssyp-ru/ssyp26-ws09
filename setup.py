from setuptools import setup, find_packages

setup(
    name="ai_agent_and_maze_evolution",
    version="0.1",
    packages=find_packages(),
    py_modules=[
        "servers.server",
        "servers.ws_inference"
    ],
)

