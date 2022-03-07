import time
import math
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)  # set seed for reproducibility


class BSDENet(torch.nn.Module):
    """
    2-layer neural network with utitlity functions for solving ODE
    """

    def __init__(
        self,
        f_fun,
        deriv_map,
        phi_fun=(lambda x: x),
        y_lo=-10.0,
        y_hi=10.0,
        x_lo=-10.0,
        x_hi=10.0,
        t_lo=0.0,
        t_hi=1.0,
        T=1.0,
        neurons=20,
        layers=5,
        bsde_lr=1e-2,
        weight_decay=0,
        batch_normalization=True,
        bsde_nb_states=1000,
        epochs=3000,
        overtrain_rate=0.1,
        device="cpu",
        bsde_activation="tanh",
        verbose=False,
        debug_gen=True,
        bsde_nb_time_intervals=5,
        **kwargs,
    ):
        super(BSDENet, self).__init__()
        self.f_fun = f_fun
        self.deriv_map = deriv_map
        self.n, self.dim = deriv_map.shape

        self.phi_fun = phi_fun
        self.znet = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.Linear(self.dim, neurons, device=device)]
                    + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
                    + [torch.nn.Linear(neurons, self.dim, device=device)]
                )
                for _ in range(bsde_nb_time_intervals)
            ]
        )
        self.z_bn_layer = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers)]
        )
        self.ynet = torch.nn.ModuleList(
            [torch.nn.Linear(self.dim, neurons, device=device)]
            + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
            + [torch.nn.Linear(neurons, 1, device=device)]
        )
        self.y_bn_layer = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers)]
        )
        self.lr = bsde_lr
        self.weight_decay = weight_decay

        x_lo, x_hi = (
            x_lo - overtrain_rate * (x_hi - x_lo),
            x_hi + overtrain_rate * (x_hi - x_lo),
        )
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.y_lo = y_lo
        self.y_hi = y_hi
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.T = T
        self.nb_time_intervals = bsde_nb_time_intervals
        self.delta_t = (T - t_lo) / bsde_nb_time_intervals

        self.nb_states = bsde_nb_states
        self.loss = torch.nn.MSELoss()  # torch.nn.HuberLoss(delta=1.0)
        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "softplus": torch.nn.ReLU(),
        }[bsde_activation]
        self.batch_normalization = batch_normalization
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.debug_gen = debug_gen

    def forward(self, x, y_or_z="y", time_index=0):
        if y_or_z == "y":
            net = self.ynet
            bn_net = self.y_bn_layer
        else:
            net = self.znet[time_index]
            # net = self.znet
            bn_net = self.z_bn_layer

        for idx, (f, bn) in enumerate(zip(net[:-1], bn_net)):
            tmp = f(x)
            tmp = self.activation(tmp)
            if self.batch_normalization:
                tmp = bn(tmp)
            if idx == 0:
                x = tmp
            else:
                # resnet
                x = tmp + x

        x = net[-1](x)
        return x

    def gen_sample(self):
        unif = torch.rand(self.nb_states * self.dim, device=self.device).reshape(
            -1, self.dim
        )
        x = [self.x_lo + (self.x_hi - self.x_lo) * unif]
        # fix all dimensions (except the first) to be a middle value for debug purposes
        if self.dim > 1 and self.debug_gen:
            x[-1][:, 1:] = (self.x_hi + self.x_lo) / 2
        dw = []

        for _ in range(self.nb_time_intervals):
            dw.append(
                math.sqrt(self.delta_t)
                * torch.randn(self.nb_states * self.dim, device=self.device).reshape(
                    -1, self.dim
                )
            )
            x.append(x[-1] + dw[-1])

        return torch.stack(dw, dim=-1), torch.stack(x, dim=-1)

    def train_and_eval(self, debug_mode=False):
        # initialize optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # lr *= .1 for every epochs // 3 steps
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[self.epochs // 3, 2 * self.epochs // 3], gamma=0.1
        )

        start = time.time()
        self.train()  # training mode
        
        # loop through epochs
        for epoch in range(self.epochs):

            dw, x = self.gen_sample()

            # clear gradients and evaluate training loss
            optimizer.zero_grad()

            y = self(x[:, :, 0], y_or_z="y")

            for t in range(self.nb_time_intervals):
                z = self(
                    x[:, :, t],
                    y_or_z="z",
                    time_index=t,
                    # torch.cat([t * torch.ones_like(x[:, :1, t]), x[:, :, t]], dim=-1),
                    # y_or_z="z",
                )
                y = (
                    y
                    - self.delta_t
                    * self.f_fun(torch.cat([y, z], dim=-1).T).unsqueeze(-1)
                    + torch.bmm(z.unsqueeze(1), dw[:, :, t].unsqueeze(-1)).squeeze(-1)
                )
                # clipping in case y explodes quickly
                y = y.clip(self.y_lo, self.y_hi)
            loss = self.loss(y.squeeze(-1), self.phi_fun(x[:, :, -1].T))

            # update model weights and record total loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 500 == 0 or epoch + 1 == self.epochs:
                if debug_mode:
                    grid = np.linspace(self.x_lo, self.x_hi, 100).astype(np.float32)
                    x_mid = (self.x_lo + self.x_hi) / 2
                    grid_nd = np.concatenate(
                        (
                            np.expand_dims(grid, axis=0),
                            x_mid * np.ones((self.dim - 1, 100)),
                        ),
                        axis=0,
                    ).astype(np.float32)
                    nn = (
                        self(
                            torch.tensor(grid_nd.T, device=self.device),
                            y_or_z="y",
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    plt.plot(grid, nn)
                    plt.show()
                    self.train()
                if self.verbose:
                    print(f"Epoch {epoch} with loss {loss.detach()}")
        if self.verbose:
            print(
                f"Training of neural network with {self.epochs} epochs take {time.time() - start} seconds."
            )


def quadratic(t, y):
    return y ** 2


def cos(t, y):
    return torch.cos(y)


def example_5(t, y):
    return t * y + y ** 2


def f_fun(y):
    # assume y[i] the (i+1)th argument of f_fun
    # return y[0] - y[0] ** 3
    return torch.exp(-y[0]) - 2 * torch.exp(-2 * y[0]) + 10 * y[1]


def phi_fun(x):
    # return -0.5 - 0.5 * torch.nn.Tanh()(-x[0] / 2)
    return torch.log(1 + x[0] ** 2)


def exact_fun(t, x, T):
    # return -0.5 - 0.5 * np.tanh(-x / 2 + 3 * (T - t) / 4)
    return np.log(1 + (x + 10 * (T - t)) ** 2)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # deriv_map maps i to the derivatives of u to the order of deriv_map[i]
    # it must be a numpy array of shape n x d, where n is the number of argument in f_fun
    # deriv_map = np.array([0]).reshape(-1, 1)
    # t_lo, T, x_lo, x_hi, patches, n = 0.0, 0.5, -8.0, 8.0, 1, 0
    deriv_map = np.array([0, 1]).reshape(-1, 1)
    t_lo, T, x_lo, x_hi, patches, n = 0.0, 0.05, -4.0, 4.0, 1, 1
    grid = np.linspace(x_lo, x_hi, 100).astype(np.float32)
    true = [exact_fun(t_lo, y, T) for y in grid]

    # Neural network with patches
    model = BSDENet(
        f_fun=f_fun,
        deriv_map=deriv_map,
        T=T,
        x_lo=x_lo,
        x_hi=x_hi,
        phi_fun=phi_fun,
        t_lo=t_lo,
        t_hi=t_lo,
        batch_normalization=True,
        patches=patches,
        verbose=True,
        nb_time_intervals=100,
    )
    model.train_and_eval(debug_mode=True)
    model.eval()
    nn = model(torch.tensor(grid, device=device).unsqueeze(-1)).detach().cpu().numpy()

    # Comparison
    plt.plot(grid, true, label="True solution")
    plt.plot(grid, nn, label=f"Neural net with {patches} patches")
    plt.legend()
    plt.show()
