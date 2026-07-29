#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <queue>
#include <unordered_map>
#include <cmath>

namespace py = pybind11;

const int INF = 10000;

const float WALL = -1.0f;
const float EMPTY = 0.0f;
const float SHADOW = 0.5f;
const float FINISH = 1.0f;

struct Offset {
    int dx, dy;

    Offset(int _dx, int _dy): dx(_dx), dy(_dy) {};
    Offset(): dx(0), dy(0) {};

    bool operator==(const Offset& other) const {
        return this->dx == other.dx && this->dy == other.dy;
    }
};

namespace std {
    template <>
    struct hash<Offset> {
        size_t operator()(const Offset& o) const noexcept {
            size_t h1 = std::hash<int>{}(o.dx);
            size_t h2 = std::hash<int>{}(o.dy);

            h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
            
            return h1;
        }
    };
}

class DiamondScanner {
private:
    std::vector<Offset> diamond_offsets;
    std::vector<Offset> diamond_border_offsets;
    std::unordered_map<Offset, int> to_diamond_vec_coord;
    std::vector<std::vector<int>> diamond_rays;
    int radius;

    void precompute_diamond_mask(int R) {
        this->diamond_offsets.clear();

        for (int x = -R; x <= R; x++)
            for (int y = R; y >= -R; y--)
                if (std::abs(x) + std::abs(y) <= R) this->diamond_offsets.push_back({x, y});
    }

    void precompute_diamond_border_mask(int R) {
        this->diamond_border_offsets.clear();

        for (int y = R; y >= -R; y--)
            for (int x = -R; x <= R; x++)
                if (std::abs(x) + std::abs(y) == R) this->diamond_border_offsets.push_back({x, y});
    }

    void precompute_offset_to_diamond_coord() {
        int idx = 0;

        for (auto offset: this->diamond_offsets)
            this->to_diamond_vec_coord[offset] = idx++; 
    }

    void precompute_rays() {
        int idx = 0;
        this->diamond_rays = std::vector<std::vector<int>>(this->diamond_border_offsets.size(), std::vector<int>());

        int x0, y0, x1, y1;
        int dx, dy, sx, sy, err;
        int e2;

        for (auto border: this->diamond_border_offsets) {
            x1 = border.dx;
            y1 = border.dy;

            x0 = y0 = 0;

            dx = std::abs(border.dx);
            dy = std::abs(border.dy);

            sx = (border.dx > 0 ? 1: -1);
            sy = (border.dy > 0 ? 1: -1);

            err = dx - dy;

            while (true) {
                this->diamond_rays[idx].push_back(this->to_diamond_vec_coord[{x0, y0}]);

                if (x0 == x1 && y0 == y1)
                    break;

                e2 = err * 2;
                if (e2 > -dy) {
                    err -= dy; 

                    x0 += sx;
                }

                if (e2 < dx) {
                    err += dx;

                    y0 += sy;
                }
            }

            idx++;
        }
    }

public:
    DiamondScanner(int Radius): radius(Radius) {
        this->precompute_diamond_mask(Radius);
        this->precompute_diamond_border_mask(Radius);
        this->precompute_offset_to_diamond_coord();
        this->precompute_rays();
    }

    size_t get_mask_size() const {
        return this->diamond_offsets.size();
    }

    py::array_t<float> get_absolute_observe(py::array_t<float> maze, int x, int y) const {
        py::array_t<float> result(this->diamond_offsets.size());
        auto res_view = result.mutable_unchecked<1>();

        auto maze_view = maze.unchecked<2>();
        
        int sx, sy, idx = 0;
        for (auto [dx, dy]: this->diamond_offsets) {
            sx = dx + x;
            sy = dy + y;

            if (sx < 0 || sx >= maze_view.shape(1) || sy < 0 || sy >= maze_view.shape(0))
                res_view(idx++) = WALL;
            else
                res_view(idx++) = maze_view(sy, sx); 
        }

        return result;
    }

    py::array_t<float> get_observe_with_shadow(py::array_t<float> maze, int x, int y) const {
        py::array_t<float> result = this->get_absolute_observe(maze, x, y);
        auto res_view = result.mutable_unchecked<1>();
        auto maze_view = maze.unchecked<2>();

        int sx, sy;
        Offset offset;
        for (const auto& ray: this->diamond_rays) {
            bool shadow_activate = false;

            for (auto ray_elm: ray) {
                if (shadow_activate) {
                    res_view(ray_elm) = SHADOW;
                    continue;
                }

                offset = this->diamond_offsets[ray_elm];
                sx = offset.dx + x;
                sy = offset.dy + y;

                if (sx < 0 || sx >= maze_view.shape(1) || sy < 0 || sy >= maze_view.shape(0)) {
                    res_view(ray_elm) = WALL;
                    shadow_activate = true;
                } else if (maze_view(sy, sx) == WALL) {
                    res_view(ray_elm) = WALL;
                    shadow_activate = true;
                }
            }
        }

        return result;
    }
    
	py::array_t<float> get_square_observe_with_shadow(py::array_t<float> maze, int x, int y) const {
		int side = 2 * this->radius + 1;

		py::array_t<float> result({side, side});
		auto res_view = result.mutable_unchecked<2>();
		auto maze_view = maze.unchecked<2>();

		std::fill_n(result.mutable_data(), result.size(), WALL);

		std::vector<bool> visible_mask(side * side, false);

		for (const auto& ray : this->diamond_rays) {
		    bool shadow_activate = false;
		
		    for (auto ray_elm : ray) {
				Offset offset = this->diamond_offsets[ray_elm];
				
				int local_row = this->radius - offset.dy;
				int local_col = offset.dx + this->radius;
				int mask_idx = local_row * side + local_col;

				int sx = offset.dx + x;
				int sy = offset.dy + y;

				if (shadow_activate) {
					if (!visible_mask[mask_idx]) {
					    res_view(local_row, local_col) = SHADOW;
					}
					
					continue;
				}

				if (sx < 0 || sx >= maze_view.shape(1) || sy < 0 || sy >= maze_view.shape(0)) {
					res_view(local_row, local_col) = WALL;
					visible_mask[mask_idx] = true;
					shadow_activate = true;
				}  else if (maze_view(sy, sx) == WALL) {
					res_view(local_row, local_col) = WALL;
					visible_mask[mask_idx] = true;
					shadow_activate = true;
				} else {
					res_view(local_row, local_col) = maze_view(sy, sx);
					visible_mask[mask_idx] = true;
				}
	    	}
	
		}

		return result;
	}

};


py::array_t<int> maze_bfs(py::array_t<float> maze, int x, int y) {
    auto maze_mat = maze.unchecked<2>();

    std::pair<int, int> start = {x, y};

    std::queue<std::pair<int, int>> q;
    q.push(start);

    py::array_t<int> result({maze_mat.shape(0), maze_mat.shape(1)});
    std::fill_n(result.mutable_data(), result.size(), INF);

    auto dist = result.mutable_unchecked<2>();

    dist(y, x) = 0;

    std::vector<std::pair<int, int>> nbs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

    std::pair<int, int> from, to;
    while (!q.empty()) {
        from = q.front();
        q.pop();

        for (const auto& nb: nbs) {
            to = {from.first + nb.first, from.second + nb.second};

            if (!(0 <= to.first && to.first < maze_mat.shape(1) && 0 <= to.second && to.second < maze_mat.shape(0)))
                continue;

            float cell_val = maze_mat(to.second, to.first);
            if ((cell_val == EMPTY || cell_val == FINISH) && dist(to.second, to.first) == INF) {
                q.push(to);
                dist(to.second, to.first) = dist(from.second, from.first) + 1;
            }
        }
    }

    return result;
}

PYBIND11_MODULE(oar_core, m) {
	py::class_<DiamondScanner>(m, "DiamondScanner")
        .def(py::init<int>(), py::arg("R"))
        .def("get_mask_size", &DiamondScanner::get_mask_size)
        .def("get_absolute_observe", &DiamondScanner::get_absolute_observe)
        .def("get_observe_with_shadow", &DiamondScanner::get_observe_with_shadow)

        .def("get_square_observe_with_shadow", &DiamondScanner::get_square_observe_with_shadow, 
             "Возвращает квадратную матрицу 11x11 с тенями под свертку.", py::arg("maze"), py::arg("x"), py::arg("y"));
    
    m.def("maze_bfs", &maze_bfs, py::arg("maze"), "Возвращает матрицу расстояний из данной клетки до всех других (свободных) в лабиринте.", py::arg("x"), py::arg("y"));
};

