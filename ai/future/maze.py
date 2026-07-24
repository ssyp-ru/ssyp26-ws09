import numpy as np
from scipy.ndimage import distance_transform_edt


class PerlinNoise:
    def __init__(self, _width: int, _height: int, _seed: int, _res_x: int, _res_y: int):
        self.width = _width
        self.height = _height

        self.seed = _seed
        self.res_x = _res_x
        self.res_y = _res_y

    @staticmethod
    def fade(t: np.ndarray):
        return t ** 3 * (t * (t * 6 - 15) + 10)

    def apply(self):
        np.random.seed(self.seed)

        angles = np.random.uniform(0, 2 * np.pi, (self.res_x + 1, self.res_y + 1))

        grid_g_x = np.cos(angles)
        grid_g_y = np.sin(angles)

        x = np.linspace(0, self.res_x, self.width, endpoint=False)
        y = np.linspace(0, self.res_y, self.height, endpoint=False)
        X, Y = np.meshgrid(x, y)

        x0 = X.astype(int)
        y0 = Y.astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        tx = X - x0
        ty = Y - y0

        g00_x, g00_y = grid_g_x[x0, y0], grid_g_y[x0, y0]
        g10_x, g10_y = grid_g_x[x1, y0], grid_g_y[x1, y0]
        g01_x, g01_y = grid_g_x[x0, y1], grid_g_y[x0, y1]
        g11_x, g11_y = grid_g_x[x1, y1], grid_g_y[x1, y1]


        dot00 = g00_x * tx + g00_y * ty
        dot10 = g10_x * (tx - 1) + g10_y * ty
        dot01 = g01_x * tx + g01_y * (ty - 1)
        dot11 = g11_x * (tx - 1) + g11_y * (ty - 1)

        u = self.fade(tx)
        v = self.fade(ty)

        nx0 = dot00 + u * (dot10 - dot00)
        nx1 = dot01 + u * (dot11 - dot01)
        perlin_map = nx0 + v * (nx1 - nx0)

        return perlin_map


class MazeMetricSpace:
    def __init__(self, _width: int, _height: int, _types_cnt: int = 2):
        self.width = _width
        self.height = _height
        self.types_cnt = _types_cnt

        self.space = np.full([_width, _height], fill_value=1, dtype=np.uint8)
        self.potentials_cache = None

    def generate_potentials_cache(self):
        self.potentials_cache = np.zeros([self.types_cnt, self.width, self.height], dtype=np.float32)

        for _type in range(self.types_cnt):
            mask = (self.space == _type)

            if np.any(mask):
                dist_mat = distance_transform_edt(~mask)

                self.potentials_cache[_type] = 1. / (dist_mat + 1.)

            else:
                self.potentials_cache[_type] = 0.

    def procedure_generate(self, seed: int, frequency: int):
        self.space = (PerlinNoise(self.width, self.height, seed, self.width // frequency, self.height // frequency).apply()
                      * self.types_cnt)

        self.generate_potentials_cache()

    def burn_corridor_step(self, x1: float, y1: float, x2: float, y2: float, val: int):
        ix1, iy1, ix2, iy2 = int(np.floor(x1)), int(np.floor(y1)), int(np.floor(x2)), int(np.floor(y2))

        step_x = lambda ix: max(0, min(ix, self.width - 1))
        step_y = lambda iy: max(0, min(iy, self.height - 1))

        ix1, iy1, ix2, iy2 = step_x(ix1), step_y(iy1), step_x(ix2), step_y(iy2)

        self.space[ix1, iy1] = val
        self.space[ix2, iy2] = val

        idx = ix2 - ix1
        idy = iy2 - iy1

        if abs(idx) == abs(idy) == 1:
            self.space[ix1, iy2] = val
            self.space[ix2, iy1] = val


class MazeTopology:
    MAX_CORRIDORS = 4096

    category_graph = np.full((MAX_CORRIDORS, 2), fill_value=-1, dtype=np.int32)
    corridors_cnt = 0

    def __init__(self, max_corridors: int):
        self.MAX_CORRIDORS = max_corridors

    def corridor_exist(self, corridor_id: int) -> bool:
        return corridor_id < self.corridors_cnt

    def add_corridor(self, _from: int = -1, _to: int = -1, mod: bool = True):
        if mod:
            _from %= self.corridors_cnt
            _to %= self.corridors_cnt

        if not self.corridor_exist(_from):
            raise RuntimeError(f"Истока коридора с данным индексом не существует! {_from} >= {self.corridors_cnt}")

        if not self.corridor_exist(_to):
            raise RuntimeError(f"Стока коридора с данным индексом не существует! {_to} >= {self.corridors_cnt}")

        self.category_graph[self.corridors_cnt] = [_from, _to]

        self.corridors_cnt += 1


class MazeParametrization:
    def __init__(self, x0: float, y0: float, maze_metric_space: MazeMetricSpace, val: int):
        self.space = maze_metric_space

        self.x_exact = float(x0)
        self.y_exact = float(y0)

        self.value = val

    def add_step(self, dx: float, dy: float):
        if dx == 0 and dy == 0:
            return

        max_projection = max(abs(float(dx)), abs(dy))
        num_sub_steps = int(np.ceil(max_projection))

        if num_sub_steps == 0:
            return

        ux = dx / num_sub_steps
        uy = dy / num_sub_steps

        for _ in range(num_sub_steps):
            x_prev, y_prev = self.x_exact, self.y_exact

            self.x_exact += ux
            self.y_exact += uy

            self.space.burn_corridor_step(x_prev, y_prev, self.x_exact, self.y_exact, self.value)


class MazeInterpolation:
    def __init__(self, metric_space: MazeMetricSpace, x_from: int, y_from: int, x_to: int = -1, y_to: int = -1,
                 tense: float = 0, charge: float = 0, materials_repulsion: np.ndarray = None):
        self.space = metric_space

        self.x0 = x_from
        self.y0 = y_from
        self.xr = x_to
        self.yr = y_to

        self.tense = tense
        self.charge = charge
        self.materials_repulsion = materials_repulsion

        self.parametrization = MazeParametrization(self.x0, self.y0, self.space, 0)

        self.params_to_calculation = np.ndarray

    def add_params_to_calculation(self, params: np.ndarray):
        self.params_to_calculation += params

    def apply(self) -> np.ndarray:
        self.space.generate_potentials_cache()

        potential = np.einsum("ijk,i->jk", self.space.potentials_cache, self.materials_repulsion)
        F = -np.gradient(potential)

        v = np.zeros([2], dtype=np.float32)
        r = np.array([self.x0, self.y0], dtype=np.float32)
        if self.xr == self.yr == -1:
            v = self.tense * v + (1 - self.tense) * F[r[0], r[1]]

            dr = v / max(1, abs(v))
            r += dr

            self.parametrization.add_step(dr[0], dr[1])

        else:
            ... # TODO: Добавить случай интерполяции между двумя фиксированными точками и обработку params_to_calculation
            	# Был план реализовать генерацию лабиринта в виде гомеоморфизма его топологии, как категорного гиперграфа
            	# на N^2 через некий метод силовых лент, основанный на принципе наименьшего действия.
            	# Также можно рассмотреть гармонические отображения.
            	# Однако в обоих случаях лучше не привязывать концы к конкретным параметрам (как это пока сделано здесь) и 
            	# позволить дочерним коридорам, в начале двигаться по родительским, а длину сделать зависимой от количества потомков.
            	# Но и это не может гарантировать отсутствие ситуаций, когда не всё умещается в двухмерном пространстве,
            	# так что можно брать срезы трёхмерного пространства или решать пересечения. В худшем случае можно
            	# позволить отображению быть лишь непрерывным.
            	# Также в случае такой реализации лабиринт может стать физически обоснованней, реалистичней и легче для Neural ODE.



class MazeHomeomorphism:
    def __init__(self, maze_topology: MazeTopology, maze_metric_space: MazeMetricSpace):
        self.topology = maze_topology
        self.space = maze_metric_space

        self.src_dst_params = np.full((self.topology.corridors_cnt, 2), fill_value=0.5, dtype=np.float32)

        self.src_dst_params = np.full((self.topology.corridors_cnt, 2), fill_value=0.5, dtype=np.float32)
        self.interpolation_params = np.full((self.topology.corridors_cnt, 2), fill_value=0, dtype=np.int8)
        self.chargers_and_tenses = np.full((self.topology.corridors_cnt, 2), fill_value=0, dtype=np.float32)

    def add_interpolation(self, corridor_id: int, tense: float, charge: float, materials_repulsion: np.ndarray, mod: bool = True):
        if mod:
            corridor_id %= self.topology.corridors_cnt

        if not self.topology.corridor_exist(corridor_id):
            raise RuntimeError("Невозможно задать интерполяцию коридора с незаданной топологией.")

        self.chargers_and_tenses[corridor_id] = [charge, tense]
        self.interpolation_params[corridor_id] = materials_repulsion

    def add_src_dst_params(self, corridor_id: int, src: float, dst: float, mod: bool = True):
        if mod:
            corridor_id %= self.topology.corridors_cnt

        if not self.topology.corridor_exist(corridor_id):
            raise RuntimeError("Невозможно задать параметры на стоке и истоке у коридора без топологии.")

        self.src_dst_params[corridor_id] = [src, dst]

    def add_homeomorphism_params(self, corridor_id: int, src: float, dst: float, tense: float, charge: float,
                                 materials_repulsion: np.ndarray, mod: bool = True):
        self.add_src_dst_params(corridor_id, src, dst, mod)
        self.add_interpolation(corridor_id, tense, charge, materials_repulsion, mod)

    def add_corridor(self, src: float, dst: float, tense: float, charge: float, materials_repulsion: np.ndarray,
                     _from: int = -1, _to: int = -1, mod: bool = True):
        self.topology.add_corridor(_from, _to)
        self.add_homeomorphism_params(self.topology.corridors_cnt, src, dst, tense, charge, materials_repulsion, mod)

    def apply(self):
        ... # TODO
