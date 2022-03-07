import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)  # set seed for reproducibility


class DGMNet(torch.nn.Module):
    """
    2-layer neural network with utitlity functions for solving ODE
    """

    def __init__(
        self,
        dgm_f_fun,
        dgm_deriv_map,
        phi_fun=(lambda x: x),
        x_lo=-10.0,
        x_hi=10.0,
        overtrain_rate=0.1,
        t_lo=0.0,
        t_hi=1.0,
        neurons=20,
        layers=5,
        dgm_lr=1e-3,
        batch_normalization=False,
        weight_decay=0,
        dgm_nb_states=1000,
        epochs=3000,
        device="cpu",
        dgm_activation="tanh",
        verbose=False,
        debug_gen=False,
        **kwargs,
    ):
        super(DGMNet, self).__init__()
        self.f_fun = dgm_f_fun
        self.n, self.dim = dgm_deriv_map.shape
        # add one more dimension of time to the left of deriv_map
        self.deriv_map = np.append(np.zeros((self.n, 1)), dgm_deriv_map, axis=-1)
        # add dt to the top of deriv_map
        self.deriv_map = np.append(
            np.array([[1] + [0] * self.dim]), self.deriv_map, axis=0
        )
        # the final deriv_map has the shape of (n + 1) x (dim + 1)

        self.phi_fun = phi_fun
        self.layer = torch.nn.ModuleList(
            [torch.nn.Linear(self.dim + 1, neurons, device=device)]
            + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
            + [torch.nn.Linear(neurons, 1, device=device)]
        )
        self.bn_layer = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(layers)]
        )
        self.lr = dgm_lr
        self.weight_decay = weight_decay

        self.loss = torch.nn.MSELoss()
        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "softplus": torch.nn.ReLU(),
        }[dgm_activation]
        self.batch_normalization = batch_normalization
        self.nb_states = dgm_nb_states
        x_lo, x_hi = (
            x_lo - overtrain_rate * (x_hi - x_lo),
            x_hi + overtrain_rate * (x_hi - x_lo),
        )
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.debug_gen = debug_gen

    def forward(self, x):
        for idx, (f, bn) in enumerate(zip(self.layer[:-1], self.bn_layer)):
            tmp = f(x)
            tmp = self.activation(tmp)
            if self.batch_normalization:
                tmp = bn(tmp)
            if idx == 0:
                x = tmp
            else:
                # resnet
                x = tmp + x

        x = self.layer[-1](x).reshape(-1)
        return x

    @staticmethod
    def nth_derivatives(order, y, x):
        for cur_dim, cur_order in enumerate(order):
            for _ in range(int(cur_order)):
                try:
                    grads = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
                except RuntimeError:
                    # when very high order derivatives are taken for polynomial function
                    # it has 0 gradient but torch has difficulty knowing that
                    # hence we handle such error separately
                    return torch.zeros_like(y)

                # update y
                y = grads[cur_dim]
        return y

    def pde_loss(self, x):
        x = x.detach().clone().requires_grad_(True)
        fun_and_derivs = []
        for order in self.deriv_map:
            fun_and_derivs.append(self.nth_derivatives(order, self(x.T), x))

        fun_and_derivs = torch.stack(fun_and_derivs)
        # recall that deriv_map has the shape of (n + 1) x (dim + 1)
        dt = fun_and_derivs[0]
        df_fun = fun_and_derivs[1:]
        return self.loss(dt + self.f_fun(df_fun), torch.zeros_like(dt))

    def gen_sample(self):
        # sample for intermediate value
        unif = torch.rand(self.nb_states, device=self.device)
        t = self.t_lo + (self.t_hi - self.t_lo) * unif
        unif = torch.rand(self.nb_states * self.dim, device=self.device).reshape(
            self.dim, -1
        )
        x = self.x_lo + (self.x_hi - self.x_lo) * unif
        tx = torch.cat((t.unsqueeze(0), x), dim=0)

        # sample for initial time, to be merged with intermediate value
        t = self.t_lo * torch.ones(self.nb_states, device=self.device)
        unif = torch.rand(self.nb_states * self.dim, device=self.device).reshape(
            self.dim, -1
        )
        x = self.x_lo + (self.x_hi - self.x_lo) * unif
        tx = torch.cat([tx, torch.cat((t.unsqueeze(0), x), dim=0)], dim=-1)

        # sample for terminal time
        t = self.t_hi * torch.ones(self.nb_states, device=self.device)
        unif = torch.rand(self.nb_states * self.dim, device=self.device).reshape(
            self.dim, -1
        )
        x = self.x_lo + (self.x_hi - self.x_lo) * unif
        tx_term = torch.cat((t.unsqueeze(0), x), dim=0)

        # fix all dimensions (except the first) to be a middle value for debug purposes
        if self.dim > 1 and self.debug_gen:
            x_mid = (self.x_hi + self.x_lo) / 2
            tx[2:, :] = x_mid
            tx_term[2:, :] = x_mid

        return tx, tx_term

    def train_and_eval(self, debug_mode=False):
        # initialize optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # lr *= .1 for every epochs // 3 steps, doesn't seem to work well for DGM
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[self.epochs // 3, 2 * self.epochs // 3], gamma=0.1
        # )

        start = time.time()
        self.train()  # training mode

        # loop through epochs
        for epoch in range(self.epochs):
            tx, tx_term = self.gen_sample()

            # clear gradients and evaluate training loss
            optimizer.zero_grad()

            # terminal loss + pde loss
            loss = self.loss(self(tx_term.T), self.phi_fun(tx_term[1:, :]))
            loss = loss + self.pde_loss(tx)

            # update model weights and record total loss
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if epoch % 500 == 0 or epoch + 1 == self.epochs:
                if debug_mode:
                    grid = np.linspace(self.x_lo, self.x_hi, 100).astype(np.float32)
                    x_mid = (self.x_lo + self.x_hi) / 2
                    grid_nd = np.concatenate(
                        (
                            self.t_lo * np.ones((1, 100)),
                            np.expand_dims(grid, axis=0),
                            x_mid * np.ones((self.dim - 1, 100)),
                        ),
                        axis=0,
                    ).astype(np.float32)
                    self.eval()
                    nn = (
                        self(torch.tensor(grid_nd.T, device=self.device))
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
    return 0.5 * y[1] + y[0] - y[0] ** 3
    # return .5 * y[2] + torch.exp(-y[0]) - 2 * torch.exp(-2 * y[0]) + 10 * y[1]


def phi_fun(x):
    return -0.5 - 0.5 * torch.nn.Tanh()(-x[0] / 2)
    # return torch.log(1 + x[0] ** 2)


def exact_fun(t, x, T):
    return -0.5 - 0.5 * np.tanh(-x / 2 + 3 * (T - t) / 4)
    # return np.log(1 + (x + 10 * (T - t)) ** 2)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Net(f_fun=f_fun, phi_fun=phi_fun, n=0)
    # # code = torch.tensor([[2,1],[2,2],[2,3],[2,4],[2,5],[3,1],[3,2],[3,3],[3,4],[3,5]]).reshape(10,1,2)
    # code = torch.tensor([[-1],[-2],[-3],[-4],[-5]]).reshape(5,1,1)
    # y0 = 2 * torch.ones(5,1)
    # print(model.code_to_function(code, y0))

    deriv_map = np.array([0, 2]).reshape(-1, 1)
    t_lo, T, x_lo, x_hi, patches, n = 0.0, 0.5, -8.0, 8.0, 1, 0
    # deriv_map = np.array([0, 1, 2]).reshape(-1, 1)
    # t_lo, T, x_lo, x_hi, patches, n = 0., .05, -4., 4., 1, 1

    grid = np.linspace(x_lo, x_hi, 100).astype(np.float32)
    xx = [[t_lo, yy] for yy in grid]
    true = [exact_fun(t_lo, y, T) for y in grid]
    terminal = [exact_fun(T, y, T) for y in grid]

    # Neural network with patches
    model = DGMNet(
        dgm_f_fun=f_fun,
        dgm_deriv_map=deriv_map,
        x_lo=x_lo,
        x_hi=x_hi,
        phi_fun=phi_fun,
        t_lo=t_lo,
        t_hi=T,
        batch_normalization=True,
        verbose=True,
    )
    # model = Net(f_fun=f_fun, t_lo=t_lo, t_hi=t_hi)
    model.train_and_eval(debug_mode=True)
    model.eval()
    nn = model(torch.tensor(xx, device=device)).detach().cpu().numpy()

    # Comparison
    plt.plot(grid, true, label="True solution")
    plt.plot(grid, terminal, label="Terminal condition")
    plt.plot(grid, nn, label=f"Deep Galerkin method")
    plt.legend()
    plt.show()
