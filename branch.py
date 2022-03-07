import time
import math
import torch
import torch.nn.functional as F
from torch.distributions.exponential import Exponential
import matplotlib.pyplot as plt
import numpy as np
import itertools
from fdb import fdb_nd

torch.manual_seed(0)  # set seed for reproducibility


# BN & dropout do not seem to help the training, so I remove the use of them
#
# what seems to work well:
# -> divide lr by 10 at the middle of the training
# -> residual network with Tanh activation
# -> overtraining so that the value is accurate even at boundaries
class Net(torch.nn.Module):
    """
    2-layer neural network with utitlity functions for solving ODE
    """

    def __init__(
        self,
        f_fun,
        deriv_map,
        phi_fun=(lambda x: x),
        x_lo=-10.0,
        x_hi=10.0,
        t_lo=0.0,
        t_hi=1.0,
        T=1.0,
        branch_exponential_lambda=1.0,
        neurons=20,
        layers=5,
        branch_lr=1e-2,
        weight_decay=0,
        branch_nb_path_per_state=10000,
        branch_nb_states=100,
        branch_nb_states_per_batch=100,
        epochs=3000,
        batch_normalization=True,
        antithetic=True,
        dydx_lam=0,
        overtrain_rate=0.1,
        device="cpu",
        branch_activation="tanh",
        verbose=False,
        debug_gen=True,
        branch_patches=1,
        outlier_percentile=1,
        **kwargs,
    ):
        super(Net, self).__init__()
        self.f_fun = f_fun
        self.deriv_map = deriv_map
        self.n, self.dim = deriv_map.shape

        # store the fdb results for quicker lookup
        start = time.time()
        self.fdb_lookup = {
            tuple(deriv): fdb_nd(self.n, tuple(deriv)) for deriv in deriv_map
        }
        self.fdb_runtime = time.time() - start
        self.mechanism_tot_len = self.dim * self.n ** 2 + sum(
            [len(self.fdb_lookup[tuple(deriv)]) for deriv in deriv_map]
        )

        self.phi_fun = phi_fun
        self.layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [torch.nn.Linear(self.dim + 1, neurons, device=device)]
                    + [
                        torch.nn.Linear(neurons, neurons, device=device)
                        for _ in range(layers)
                    ]
                    + [torch.nn.Linear(neurons, 1, device=device)]
                )
                for _ in range(branch_patches)
            ]
        )
        self.bn_layer = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [
                        torch.nn.BatchNorm1d(neurons, device=device)
                        for _ in range(layers)
                    ]
                )
                for _ in range(branch_patches)
            ]
        )
        self.lr = branch_lr
        self.weight_decay = weight_decay
        self.dropout = torch.nn.Dropout(p=0.1)

        self.loss = torch.nn.MSELoss()
        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "softplus": torch.nn.ReLU(),
        }[branch_activation]
        self.batch_normalization = batch_normalization
        self.nb_states = branch_nb_states
        self.nb_states_per_batch = branch_nb_states_per_batch
        self.nb_path_per_state = branch_nb_path_per_state
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.adjusted_x_boundaries = (
            x_lo - overtrain_rate * (x_hi - x_lo),
            x_hi + overtrain_rate * (x_hi - x_lo),
        )
        self.t_lo = t_lo
        self.t_hi = t_hi
        self.T = T
        self.delta_t = (T - t_lo) / branch_patches
        self.outlier_percentile = outlier_percentile

        self.exponential_lambda = branch_exponential_lambda
        self.epochs = epochs
        self.dydx_lam = dydx_lam
        self.antithetic = antithetic
        self.device = device
        self.verbose = verbose
        self.debug_gen = debug_gen
        self.patches = branch_patches
        self.t_boundaries = torch.tensor(
            ([t_lo + i * self.delta_t for i in range(branch_patches)] + [T])[::-1],
            device=device,
        )
        if t_lo == t_hi:
            self.adjusted_t_boundaries = [
                (lo, hi) for hi, lo in zip(self.t_boundaries[1:], self.t_boundaries[1:])
            ]
        else:
            # is this even needed???
            # over-train by 10% of the range so that the performance is good even at the actual boundary
            self.adjusted_t_boundaries = [
                (
                    lo - overtrain_rate * (hi - lo),
                    min(T, hi + overtrain_rate * (hi - lo)),
                )
                for hi, lo in zip(self.t_boundaries[:-1], self.t_boundaries[1:])
            ]

    def forward(self, x, patch=None):
        if patch is not None:
            y = x
            for idx, (f, bn) in enumerate(
                zip(self.layer[patch][:-1], self.bn_layer[patch])
            ):
                tmp = f(y)
                tmp = self.activation(tmp)
                if self.batch_normalization:
                    tmp = bn(tmp)
                # tmp = self.dropout(tmp)
                if idx == 0:
                    y = tmp
                else:
                    # resnet
                    y = tmp + y

            y = self.layer[patch][-1](y).reshape(-1)
        else:
            yy = []
            for p in range(self.patches):
                y = x
                for idx, (f, bn) in enumerate(
                    zip(self.layer[p][:-1], self.bn_layer[p])
                ):
                    tmp = f(y)
                    tmp = self.activation(tmp)
                    if self.batch_normalization:
                        tmp = bn(tmp)
                    # tmp = self.dropout(tmp)
                    if idx == 0:
                        y = tmp
                    else:
                        # resnet
                        y = tmp + y
                yy.append(self.layer[p][-1](y).reshape(-1))
            idx = self.bisect_left(x[:, 0])
            y = torch.gather(torch.stack(yy, dim=-1), -1, idx.reshape(-1, 1)).squeeze(
                -1
            )
        return y

    def bisect_left(self, val):
        idx = (
            torch.max(self.t_boundaries <= (val + 1e-8).reshape(-1, 1), dim=1)[
                1
            ].reshape(val.shape)
            - 1
        )
        # t_boundaries[0], use the first network
        idx = idx.where(~(val == self.t_boundaries[0]), torch.zeros_like(idx))
        # t_boundaries[-1], use the last network
        idx = idx.where(
            ~(val == self.t_boundaries[-1]),
            (self.t_boundaries.shape[0] - 2) * torch.ones_like(idx),
        )
        return idx

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

    def adjusted_phi(self, x, T, patch):
        if patch == 0:
            return self.phi_fun(x)
        else:
            self.eval()
            xx = torch.stack((T.reshape(-1), x.reshape(-1)), dim=-1)
            return self(xx, patch=patch - 1).reshape(-1, self.nb_path_per_state)

    def code_to_function(self, code, x, T, patch=0):
        # shape of x -> d x nb_states x nb_paths_per_state
        # shape of fun_val -> nb_states x nb_paths_per_state
        x = x.detach().clone().requires_grad_(True)
        fun_val = torch.zeros_like(x[0])

        # code (neg_num_1, ..., neg_num_d) -> d/dx1^{-neg_num_1 - 1} ... d/dxd^{-neg_num_d - 1} phi(x1, ..., xd)
        if code[0] < 0:
            return self.nth_derivatives(
                -code - 1, self.adjusted_phi(x, T, patch), x
            ).detach()

        # code (pos_num_0, ..., pos_num_n) -> d/dx0^{pos_num_0 - 1} ... d/dxn^{pos_num_n - 1} f(x0, x1, ..., xn)
        if code[0] > 0:
            y = []
            for order in self.deriv_map:
                y.append(
                    self.nth_derivatives(
                        order, self.adjusted_phi(x, T, patch), x
                    ).detach()
                )
            y = torch.stack(y[: self.n]).requires_grad_()

            return self.nth_derivatives(code - 1, self.f_fun(y), y).detach()

        return fun_val

    def gen_bm(self, dt, nb_states):
        dt = dt.clip(min=0.0)  # so that we can safely take square root of dt

        if self.antithetic:
            # antithetic variates
            dw = torch.sqrt(dt) * torch.randn(
                self.dim, nb_states, self.nb_path_per_state // 2, device=self.device
            ).repeat(1, 1, 2)
            dw[:, :, : (self.nb_path_per_state // 2)] *= -1
        else:
            # usual generation
            dw = torch.sqrt(dt) * torch.randn(
                self.dim, nb_states, self.nb_path_per_state, device=self.device
            )
        return dw

    def gen_sample_batch(self, t, T, x, mask, H, code, patch):
        """
        shape of t    -> nb_states x nb_paths_per_state
        shape of T    -> nb_states x nb_paths_per_state
        shape of x    -> d x nb_states x nb_paths_per_state
        shape of mask -> nb_states x nb_paths_per_state
        shape of H    -> nb_states x nb_paths_per_state
        shape of code -> d for negative code
                      -> n + 1 for positive code
        """
        # return zero tensor when no operation is needed
        ans = torch.zeros_like(t)
        if ~mask.any():
            return ans

        nb_states, _ = t.shape
        tau = Exponential(
            self.exponential_lambda
            * torch.ones(nb_states, self.nb_path_per_state, device=self.device)
        ).sample()
        dw = self.gen_bm(T - t, nb_states)

        ############################### for t + tau >= T
        mask_now = mask.bool() * (t + tau >= T)
        if mask_now.any():
            tmp = (
                H[mask_now]
                * self.code_to_function(code, (x + dw)[:, mask_now], T[mask_now], patch)
                / torch.exp(-self.exponential_lambda * (T - t)[mask_now])
            )
            ans[mask_now] = tmp

        ############################### for t + tau < T
        dw = self.gen_bm(tau, nb_states)
        mask_now = mask.bool() * (t + tau < T)

        # return when all processes die
        if ~mask_now.any():
            return ans

        # uniform distribution to choose from the set of mechanism
        unif = torch.rand(nb_states, self.nb_path_per_state, device=self.device)

        # identity code (-1, ..., -1)
        if (len(code) == self.dim) and (code == [-1] * self.dim).all():
            tmp = self.gen_sample_batch(
                t + tau,
                T,
                x + dw,
                mask_now,
                H / self.exponential_lambda / torch.exp(-self.exponential_lambda * tau),
                np.array([1] * self.n),
                patch,
            )
            ans = ans.where(~mask_now, tmp)

        # negative code
        elif code[0] < 0:
            order = tuple(-code - 1)
            # if c is not in the lookup, add it
            if order not in self.fdb_lookup.keys():
                start = time.time()
                self.fdb_lookup[order] = fdb_nd(self.n, order)
                self.fdb_runtime += (time.time() - start)
            L = self.fdb_lookup[order]
            idx = (unif * len(L)).long()
            idx_counter = 0

            # loop through all fdb elements
            for fdb in L:
                mask_tmp = mask_now * (idx == idx_counter)
                A = fdb.coeff * self.gen_sample_batch(
                    t + tau,
                    T,
                    x + dw,
                    mask_tmp,
                    len(L)
                    * H
                    / self.exponential_lambda
                    / torch.exp(-self.exponential_lambda * tau),
                    np.array(fdb.lamb) + 1,
                    patch,
                )

                for ll, k_arr in fdb.l_and_k.items():
                    for q in range(self.n):
                        for _ in range(k_arr[q]):
                            dw = self.gen_bm(tau, nb_states)
                            A = A * self.gen_sample_batch(
                                t + tau,
                                T,
                                x + dw,
                                mask_tmp,
                                torch.ones_like(t),
                                -self.deriv_map[q] - ll - 1,
                                patch,
                            )
                ans = ans.where(~mask_tmp, A)
                idx_counter += 1

        # positive code
        elif code[0] > 0:
            idx = (unif * self.mechanism_tot_len).long()
            idx_counter = 0

            # positive code part 1
            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.dim):
                        mask_tmp = mask_now * (idx == idx_counter)
                        code_increment = np.zeros_like(self.deriv_map[i])
                        code_increment[k] += 1
                        A = self.gen_sample_batch(
                            t + tau,
                            T,
                            x + dw,
                            mask_tmp,
                            torch.ones_like(t),
                            -self.deriv_map[i] - code_increment - 1,
                            patch,
                        )
                        dw = self.gen_bm(tau, nb_states)
                        B = self.gen_sample_batch(
                            t + tau,
                            T,
                            x + dw,
                            mask_tmp,
                            torch.ones_like(t),
                            -self.deriv_map[j] - code_increment - 1,
                            patch,
                        )
                        # only code + 1 in the dimension j and l
                        code_increment = np.zeros_like(code)
                        code_increment[i] += 1
                        code_increment[j] += 1
                        dw = self.gen_bm(tau, nb_states)
                        tmp = self.gen_sample_batch(
                            t + tau,
                            T,
                            x + dw,
                            mask_tmp,
                            -0.5
                            * self.mechanism_tot_len
                            * A
                            * B
                            * H
                            / self.exponential_lambda
                            / torch.exp(-self.exponential_lambda * tau),
                            code + code_increment,
                            patch,
                        )
                        ans = ans.where(~mask_tmp, tmp)
                        idx_counter += 1

            # positive code part 2
            for k in range(self.n):
                for fdb in self.fdb_lookup[tuple(self.deriv_map[k])]:
                    mask_tmp = mask_now * (idx == idx_counter)
                    A = fdb.coeff * self.gen_sample_batch(
                        t + tau,
                        T,
                        x + dw,
                        mask_tmp,
                        torch.ones_like(t),
                        np.array(fdb.lamb) + 1,
                        patch,
                    )
                    for ll, k_arr in fdb.l_and_k.items():
                        for q in range(self.n):
                            for _ in range(k_arr[q]):
                                dw = self.gen_bm(tau, nb_states)
                                A = A * self.gen_sample_batch(
                                    t + tau,
                                    T,
                                    x + dw,
                                    mask_tmp,
                                    torch.ones_like(t),
                                    -self.deriv_map[q] - ll - 1,
                                    patch,
                                )
                    code_increment = np.zeros_like(code)
                    code_increment[k] += 1
                    dw = self.gen_bm(tau, nb_states)
                    tmp = self.gen_sample_batch(
                        t + tau,
                        T,
                        x + dw,
                        mask_tmp,
                        self.mechanism_tot_len
                        * A
                        * H
                        / self.exponential_lambda
                        / torch.exp(-self.exponential_lambda * tau),
                        code + code_increment,
                        patch,
                    )
                    ans = ans.where(~mask_tmp, tmp)
                    idx_counter += 1
        return ans

    def gen_sample(self, patch, t=None):
        if t is None:
            nb_states = self.nb_states
        else:
            nb_states, _ = t.shape
        states_per_batch = min(nb_states, self.nb_states_per_batch)
        batches = math.ceil(nb_states / states_per_batch)
        t_lo, t_hi = self.adjusted_t_boundaries[patch]
        x_lo, x_hi = self.adjusted_x_boundaries
        xx, yy, dydx = [], [], []
        for _ in range(batches):
            unif = (
                torch.rand(states_per_batch, device=self.device)
                .repeat(self.nb_path_per_state)
                .reshape(self.nb_path_per_state, -1)
                .T
            )
            t = t_lo + (t_hi - t_lo) * unif
            unif = (
                torch.rand(self.dim * states_per_batch, device=self.device)
                .repeat(self.nb_path_per_state)
                .reshape(self.nb_path_per_state, -1)
                .T.reshape(self.dim, states_per_batch, self.nb_path_per_state)
            )
            x = x_lo + (x_hi - x_lo) * unif
            # fix all dimensions (except the first) to be a middle value for debug purposes
            if self.dim > 1 and self.debug_gen:
                x[1:, :, :] = (x_hi + x_lo) / 2
            T = (t_lo + self.delta_t) * torch.ones_like(t)
            # xx.append(torch.cat((t[:, :, 0], x[:, :, 0]), dim=0).T.detach())
            xx.append(torch.cat((t[:, :1], x[:, :, 0].T), dim=-1).detach())
            yy_tmp = self.gen_sample_batch(
                t,
                T,
                x,
                torch.ones_like(t),
                torch.ones_like(t),
                np.array([-1] * self.dim),
                patch,
            ).detach()
            # let (lo, hi) be 
            # (self.outlier_percentile, 100 - self.outlier_percentile)
            # percentile of yy_tmp
            #
            # set the boundary as [lo-1000*(hi-lo), hi+1000*(hi-lo)]
            # samples out of this boundary is considered as outlier and removed
            lo, hi = (
                yy_tmp.nanquantile(self.outlier_percentile/100, dim=1, keepdim=True), 
                yy_tmp.nanquantile(1 - self.outlier_percentile/100, dim=1, keepdim=True)
            )
            lo, hi = lo - 1000 * (hi - lo), hi + 1000 * (hi - lo)
            mask = torch.logical_or(
                torch.logical_and(lo <= yy_tmp, yy_tmp <= hi), yy_tmp.isnan()
            )
            yy.append((yy_tmp.nan_to_num() * mask).sum(dim=1) / mask.sum(dim=1))
            # # debug of samples outlier
            # idx = yy[-1].argmax()
            # print(f"The max value: x = {x[0, idx, 0]}; y = {yy[-1][idx]}")
            # plt.plot(yy_tmp[idx].cpu(), "+", label="Distribution of samples producing max mean.")
            # plt.show()

            if self.dydx_lam > 0:
                # for the derivatives of solution
                dydx.append(
                    self.gen_sample_batch(
                        t,
                        T,
                        x,
                        torch.ones_like(t),
                        torch.ones_like(t),
                        np.array([-2] * self.dim),
                        patch,
                    )
                    .mean(dim=1)
                    .detach()
                )

        return (
            torch.cat(xx),
            torch.cat(yy),
            torch.cat(dydx) if self.dydx_lam > 0 else None,
        )

    def train_and_eval(self, debug_mode=False):
        for p in range(self.patches):
            # initialize optimizer
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            # lr *= .1 for every epochs // 3 steps
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[self.epochs // 3, 2 * self.epochs // 3],
                gamma=0.1,
            )

            start = time.time()
            x, y, dydx = self.gen_sample(patch=p)
            x = x.requires_grad_(dydx is not None)
            if self.verbose:
                print(
                    f"Patch {p}: generation of samples take {time.time() - start} seconds."
                )

            start = time.time()
            self.train()  # training mode

            # loop through epochs
            for epoch in range(self.epochs):
                # clear gradients and evaluate training loss
                optimizer.zero_grad()
                predict = self(x, patch=p)
                loss = self.loss(predict, y)
                if dydx is not None:
                    loss = loss + self.dydx_lam * self.loss(
                        torch.autograd.grad(predict.sum(), x, create_graph=True)[0][
                            :, 1
                        ],
                        dydx,
                    )
                # update model weights and record total loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                if epoch % 500 == 0 or epoch + 1 == self.epochs:
                    if debug_mode:
                        grid = np.linspace(self.x_lo, self.x_hi, 100)
                        x_mid = (self.x_lo + self.x_hi) / 2
                        t_lo = x[:, 0].min().item()
                        grid_nd = np.concatenate(
                            (
                                t_lo * np.ones((1, 100)),
                                np.expand_dims(grid, axis=0),
                                x_mid * np.ones((self.dim - 1, 100)),
                            ),
                            axis=0,
                        ).astype(np.float32)
                        self.eval()
                        nn = (
                            self(torch.tensor(grid_nd.T, device=self.device), patch=p)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        f = plt.figure()
                        plt.plot(x[:, 1].detach().cpu(), y.cpu(), "+", label="Monte Carlo samples")
                        plt.plot(grid, nn, label="Neural network function")
                        plt.legend()
                        plt.show()
                        f.savefig("plot/debug.pdf", bbox_inches="tight")
                        # save points to csv
                        data = np.stack((x[:, 1].detach().cpu().numpy(), y.cpu().numpy()), axis=-1)
                        np.savetxt(
                            "log/debug_mc_samples.csv",
                            data,
                            delimiter=",",
                            header="x,y",
                            comments="",
                        )
                        data = np.stack((grid, nn), axis=-1)
                        np.savetxt(
                            "log/debug_nn.csv",
                            data,
                            delimiter=",",
                            header="x,nn",
                            comments="",
                        )
                        if dydx is not None:
                            xx = torch.tensor(
                                [[t_lo, yy] for yy in grid],
                                device=self.device,
                                requires_grad=True,
                            )
                            nn = (
                                torch.autograd.grad(self(xx, patch=p).sum(), xx)[0][
                                    :, 1
                                ]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            plt.plot(grid, nn)
                            plt.plot(x[:, 1].detach().cpu(), dydx.cpu(), "+")
                            plt.show()
                        self.train()
                    if self.verbose:
                        print(f"Patch {p}: epoch {epoch} with loss {loss.detach()}")
            if self.verbose:
                print(
                    f"Patch {p}: training of neural network with {self.epochs} epochs take {time.time() - start} seconds."
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
    # return -.5 * y[1] + y[2] * y[0]**3
    # return -0.5 * y[2] + 10 * y[1] + y[2] / (1 + y[0] ** 2) - 2 * y[0]
    # return -.5 * y[2] + 10 * y[1] + y[0] - (y[2] / 12) ** 2 + torch.cos(math.pi * y[3] / 24)
    # return -.5 * y[1] + 5 * y[0] + torch.log(y[1]**2 + y[2]**2)


def phi_fun(x):
    # return -0.5 - 0.5 * torch.nn.Tanh()(-x[0] / 2)
    return torch.log(1 + x[0] ** 2)
    # return torch.sign(x[0]) * (6 * torch.abs(x[0]))**(2/3)
    # return torch.tan(x[0])
    # return x[0]**4 + x[0]**3 - 36/47 * x[0]**2 - 24*36/47 * x[0] + 4 * (36/47)**2
    # return torch.cos(x[0])


def exact_fun(t, x, T):
    # return -0.5 - 0.5 * np.tanh(-x / 2 + 3 * (T - t) / 4)
    return np.log(1 + (x + 10 * (T - t)) ** 2)
    # xx = 6 * (x + 16 * (T-t))
    # return np.sign(xx) * np.abs(xx)**(2/3)
    # return np.tan(x + 10 * (T - t))
    # xx = x + 10 * (T - t)
    # return xx**4 + xx**3 - 36/47 * xx**2 - 24*36/47 * xx + 4 * (36/47)**2
    # return np.cos(x + 5 * (T-t))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Net(f_fun=f_fun, phi_fun=phi_fun, n=3)
    # # code = torch.tensor([[-1,-1,-1,-1],[-2,-2,-2,-2],[-3,-3,-3,-3]]).reshape(3,1,4)
    # code = torch.tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]]).reshape(3,1,4)
    # x = 1e-2 * torch.ones(1,3,1)
    # # model = Net(f_fun=f_fun, phi_fun=phi_fun, n=1)
    # # code = torch.tensor([[1,1],[1,2],[1,3],[1,4],[1,5],[2,1],[2,2],[2,3],[2,4],[2,5]]).reshape(10,1,2)
    # # x = 2 * torch.ones(10,1)
    # print(model.code_to_function(code, x, T=model.T))

    # deriv_map maps i to the derivatives of u to the order of deriv_map[i]
    # it must be a numpy array of shape n x d, where n is the number of argument in f_fun
    # deriv_map = np.array([0]).reshape(-1, 1)
    # t_lo, T, x_lo, x_hi, patches, n = 0.0, 0.5, -8.0, 8.0, 1, 0
    deriv_map = np.array([0, 1]).reshape(-1, 1)
    t_lo, T, x_lo, x_hi, patches, n = 0.0, 0.05, -4.0, 4.0, 1, 1
    # deriv_map = np.array([0, 2, 3]).reshape(-1, 1)
    # t_lo, T, x_lo, x_hi, patches, n = 0., .01, 5., 6., 1, 3
    # deriv_map = np.array([0, 1, 2]).reshape(-1, 1)
    # t_lo, T, x_lo, x_hi, patches, n = 0.0, 0.01, 0.0, 1.0, 1, 2
    # deriv_map = np.array([0, 1, 2, 4]).reshape(-1, 1)
    # t_lo, T, x_lo, x_hi, patches, n = 0.0, 0.04, -5., 5., 1, 4
    # deriv_map = np.array([1, 2, 3]).reshape(-1, 1)
    # t_lo, T, x_lo, x_hi, patches, n = 0.0, 0.02, -3., 3., 1, 3
    grid = np.linspace(x_lo, x_hi, 100).astype(np.float32)
    xx = [[t_lo, yy] for yy in grid]
    true = [exact_fun(t_lo, y, T) for y in grid]
    terminal = [exact_fun(T, y, T) for y in grid]

    # Neural network with patches
    model = Net(
        f_fun=f_fun,
        deriv_map=deriv_map,
        T=T,
        x_lo=x_lo,
        x_hi=x_hi,
        phi_fun=phi_fun,
        t_lo=t_lo,
        t_hi=t_lo,
        batch_normalization=True,
        branch_patches=patches,
        # branch_nb_path_per_state=100000,
        branch_nb_states=100,
        # overtrain_rate=.0,
        verbose=True,
        branch_exponential_lambda=-math.log(0.95) / T,
        # branch_exponential_lambda=.01,
        branch_lr=1e-2,
        antithetic=False,
        epochs=3000,
        # dydx_lam=10,
    )
    model.train_and_eval(debug_mode=True)
    model.eval()
    nn = model(torch.tensor(xx, device=device)).detach().cpu().numpy()

    # Comparison
    plt.plot(grid, true, label="True solution")
    plt.plot(grid, terminal, label="Terminal condition")
    plt.plot(grid, nn, label=f"Neural net with {patches} patches")
    plt.legend()
    plt.show()
