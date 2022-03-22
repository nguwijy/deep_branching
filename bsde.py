import time
import math
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)  # set seed for reproducibility


class BSDENet(torch.nn.Module):
    """
    deep BSDE approach to solve PDE with utility functions
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
        fix_all_dim_except_first=True,
        bsde_nb_time_intervals=4,
        second_order=False,
        **kwargs,
    ):
        super(BSDENet, self).__init__()
        # we assume f only depends on the diagonal entry of Gamma
        # hence, idx 0 -> y; idx 1 to d -> z; idx d+1 to 2d -> Gamma
        self.f_fun = f_fun
        self.deriv_map = deriv_map
        self.n, self.dim = deriv_map.shape

        self.phi_fun = phi_fun
        self.second_order = second_order
        if second_order:
            # second order BSDE
            self.anet = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [torch.nn.Linear(self.dim, neurons, device=device)]
                        + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
                        + [torch.nn.Linear(neurons, self.dim, device=device)]
                    )
                    for _ in range(bsde_nb_time_intervals)
                ]
            )
            self.a_bn_layer = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [torch.nn.BatchNorm1d(self.dim, device=device)]
                        + [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers + 1)]
                        + [torch.nn.BatchNorm1d(self.dim, device=device)]
                    )
                    for _ in range(bsde_nb_time_intervals)
                ]
            )
            self.gnet = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [torch.nn.Linear(self.dim, neurons, device=device)]
                        + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
                        + [torch.nn.Linear(neurons, self.dim, device=device)]
                    )
                    for _ in range(bsde_nb_time_intervals)
                ]
            )
            self.g_bn_layer = torch.nn.ModuleList(
                [
                    torch.nn.ModuleList(
                        [torch.nn.BatchNorm1d(self.dim, device=device)]
                        + [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers + 1)]
                        + [torch.nn.BatchNorm1d(self.dim, device=device)]
            )
                    for _ in range(bsde_nb_time_intervals)
                ]
            )
            self.znet = torch.nn.ModuleList(
                        [torch.nn.Linear(self.dim, neurons, device=device)]
                        + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
                        + [torch.nn.Linear(neurons, self.dim, device=device)]
                    )
            self.z_bn_layer = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(self.dim, device=device)]
                + [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers + 1)]
                + [torch.nn.BatchNorm1d(self.dim, device=device)]
            )
            self.ynet = torch.nn.ModuleList(
                [torch.nn.Linear(self.dim, neurons, device=device)]
                + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
                + [torch.nn.Linear(neurons, 1, device=device)]
            )
            self.y_bn_layer = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(self.dim, device=device)]
                + [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers + 1)]
                + [torch.nn.BatchNorm1d(1, device=device)]
            )
        else:
            # classical BSDE
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
                [
                    torch.nn.ModuleList(
                        [torch.nn.BatchNorm1d(self.dim, device=device)]
                        + [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers + 1)]
                        + [torch.nn.BatchNorm1d(self.dim, device=device)]
                    )
                    for _ in range(bsde_nb_time_intervals)
                ]
            )
            self.ynet = torch.nn.ModuleList(
                [torch.nn.Linear(self.dim, neurons, device=device)]
                + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
                + [torch.nn.Linear(neurons, 1, device=device)]
            )
            self.y_bn_layer = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(self.dim, device=device)]
                + [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers + 1)]
                + [torch.nn.BatchNorm1d(1, device=device)]
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
        self.fix_all_dim_except_first = fix_all_dim_except_first

    def forward(self, x, variable="y", time_index=0):
        """
        self(x) evaluates the neural network approximation NN(x)

        classical BSDE has two kinds of network,
        1. y for the approximation of the true solution at initial time t_lo
        2. z for the approximation of the first order derivatives of true solution from time t_lo to t_hi

        second order BSDE has four kinds of network,
        1. y for the approximation of the true solution at initial time t_lo
        2. z for the approximation of the first order derivatives of true solution at initial time t_lo
        3. gamma for the approximation of the second order derivatives of true solution from time t_lo to t_hi
        4. A for the approximation of the LDu from time t_lo to t_hi
        """
        if variable == "y":
            net = self.ynet
            bn_net = self.y_bn_layer
        elif variable == "z":
            net = self.znet if self.second_order else self.znet[time_index]
            bn_net = self.z_bn_layer if self.second_order else self.z_bn_layer[time_index]
        elif variable == "g":
            net = self.gnet[time_index]
            bn_net = self.g_bn_layer[time_index]
        elif variable == "a":
            net = self.anet[time_index]
            bn_net = self.a_bn_layer[time_index]

        if self.batch_normalization:
            x = bn_net[0](x)

        for idx, (f, bn) in enumerate(zip(net[:-1], bn_net[1:-1])):
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
        """
        generate brownian motion sqrt{dt} x Gaussian from time t_lo to t_hi
        """
        unif = torch.rand(self.nb_states * self.dim, device=self.device).reshape(
            -1, self.dim
        )
        x = [self.x_lo + (self.x_hi - self.x_lo) * unif]
        # fix all dimensions (except the first) to be the middle value
        if self.dim > 1 and self.fix_all_dim_except_first:
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
        """
        generate sample and evaluate (plot) NN approximation when debug_mode=True
        """
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

            y = self(x[:, :, 0], variable="y")
            if self.second_order:
                z = self(x[:, :, 0], variable="z")

            for t in range(self.nb_time_intervals):
                if self.second_order:
                    # second order BSDE
                    gamma = self(
                        x[:, :, t],
                        variable="g",
                        time_index=t,
                    )
                    a = self(
                        x[:, :, t],
                        variable="a",
                        time_index=t,
                    )
                    f_input = torch.cat([y, z, gamma], dim=-1).T
                else:
                    # classical BSDE
                    z = self(
                        x[:, :, t],
                        variable="z",
                        time_index=t,
                    )
                    f_input = torch.cat([y, z], dim=-1).T
                y = (
                    y
                    - self.delta_t
                    * self.f_fun(f_input).unsqueeze(-1)
                    + torch.bmm(z.unsqueeze(1), dw[:, :, t].unsqueeze(-1)).squeeze(-1)
                )
                if self.second_order:
                    z = (
                            z
                            + self.delta_t * a
                            + dw[:, :, t] * gamma
                    )
                # clipping in case y explodes quickly
                y = y.clip(self.y_lo, self.y_hi)
            loss = self.loss(y.squeeze(-1), self.phi_fun(x[:, :, -1].T))

            # update model weights and schedule
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print loss information every 500 epochs
            if epoch % 500 == 0 or epoch + 1 == self.epochs:
                if debug_mode:
                    self.eval()
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
                            variable="y",
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
        self.eval()


if __name__ == "__main__":
    # configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, x_lo, x_hi, dim = .05, -4.0, 4.0, 3
    # deriv_map is n x d array defining lambda_1, ..., lambda_n
    deriv_map = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    def f_fun(y):
        """
        idx 0      -> no deriv
        idx 1 to d -> first deriv
        """
        return y[1:].sum(dim=0) + dim * torch.exp(-y[0]) * (1 - 2 * torch.exp(-y[0]))

    def g_fun(x):
        return torch.log(1 + x.sum(dim=0) ** 2)


    # initialize model and training
    model = BSDENet(
        deriv_map=deriv_map,
        f_fun=f_fun,
        phi_fun=g_fun,
        t_hi=T,
        T=T,
        x_lo=x_lo,
        x_hi=x_hi,
        device=device,
        verbose=True,
        second_order=True,
    )
    model.train_and_eval()


    # define exact solution and plot the graph
    def exact_fun(t, x, T):
        return np.log(1 + (x.sum(axis=0) + dim * (T - t)) ** 2)

    grid = torch.linspace(x_lo, x_hi, 100).unsqueeze(dim=-1)
    nn_input = torch.cat((grid, torch.zeros((100, 2))), dim=-1)
    plt.plot(grid, model(nn_input).detach(), label="Deep BSDE")
    plt.plot(grid, exact_fun(0, nn_input.numpy().T, T), label="True solution")
    plt.legend()
    plt.show()
