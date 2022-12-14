import numpy as np


class Hopfield():
    def __init__(self, data_shape_x: int, data_shape_y: int):
        self._x = data_shape_x
        self._y = data_shape_y

    def sgn(self, step_d: np.ndarray, d: np.ndarray):
        step_d[step_d > 0] = 1
        step_d[step_d < 0] = -1
        step_d[step_d == 0] = d[step_d == 0]  # select pervious step state if new step state is 0
        return step_d

    def get_data(self, file_name: str) -> np.ndarray:
        with open(f"Hopfield_dataset/{file_name}", "r") as f:
            lines = f.readlines()
        lines = [1 if c == '1' else -1 for x in lines for c in x.strip('\n')]  # '1' => 1 and ' ' => -1
        xs = np.reshape(lines, (-1, self._x*self._y))  # each pic is one row (data_count, data_x*data_y)
        return xs

    def make_weights(self, xs: np.ndarray): # w[i][j] = sum(each row's i * each row's j), not dot!
        x_len = xs.shape[1]
        ws = np.zeros((x_len, x_len), dtype=int)
        for i in range(x_len):
            for j in range(x_len):
                if i == j:
                    continue
                ws[i, j] = np.sum(xs[:, i] * xs[:, j])
        return ws

    def train(self, mode: str):
        xs = self.get_data(f"{mode}_Training.txt")
        ws = self.make_weights(xs)
        ys = self.get_data(f"{mode}_Testing.txt")
        self._xs = xs # data for training
        self._ws = ws # model weights
        self._ys = ys # data for testing
        return self

    def step(self, d: np.ndarray):
        return self.sgn(self._ws @ d.T, d)

    def print_d(self, d: np.ndarray, desc=""):
        reshaped = d.reshape(self._y, self._x)
        print(desc, reshaped, sep="\n")
        return reshaped

    def test_steps(self, x: np.ndarray, step_limit=10):
        yield x
        for _ in range(step_limit):
            new_x = self.step(x)
            if np.array_equal(new_x, x):
                break
            x = new_x
            yield x

    def get_noise_data(self, p=0.25): # p is the probability of flipping a bit, 1=>-1 or -1=>1
        ans_x = self._xs[np.random.choice(self._xs.shape[0], 1)[0]] # as answer
        noisy_x = ans_x*np.random.choice([-1, 1], self._x*self._y, p=[p, 1-p]) # as noisy data
        return ans_x, noisy_x


if __name__ == "__main__":
    # Train
    model = Hopfield(9, 12)
    model.train("Basic")

    # Test
    for class_i, x in enumerate(model._ys):
        print(f"Class {class_i}")
        for step_i, step in enumerate(model.test_steps(x)):
            model.print_d(step, f"Step {step_i}")

    # Noisy test
    ans, noisy_data = model.get_noise_data()
    print("Noisy data")
    model.print_d(ans, "Correct")
    for i, s in enumerate(model.test_steps(noisy_data)):
        model.print_d(s, f"Step {i}")
