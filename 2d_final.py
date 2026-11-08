import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
from torch.func import jvp, functional_call

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

GMM_WEIGHTS = torch.tensor([0.40, 0.30, 0.30], device=DEVICE)
GMM_MEANS = torch.tensor([
    [0.10,  0.15],
    [3.30,  2.30],
    [3.45, -2.50],
], device=DEVICE)

GMM_COVS = torch.tensor([
    [[ 0.85,  0.55], [ 0.55,  0.65]],
    [[ 0.95, -0.55], [-0.55,  0.50]],
    [[ 0.35,  0.00], [ 0.00,  0.85]],
], device=DEVICE)

# Measurement Line passing through (-4, 1) and (4.8, -4.5):
Q_MAT = torch.tensor([[0.625, 1.0]], device=DEVICE)
q_VAL = torch.tensor([-1.5], device=DEVICE)
R_COV = torch.tensor([[1e-4]], device=DEVICE)

def generate_initial_grid():
    """Exact 5x5 grid in [-2, 2] x [-2, 2] matching paper."""
    x_vals = torch.linspace(-2.0, 2.0, 5, device=DEVICE)
    y_vals = torch.linspace(2.0, -2.0, 5, device=DEVICE)
    grid_y, grid_x = torch.meshgrid(y_vals, x_vals, indexing='ij')
    return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

def sample_gmm(n_samples):
    cat = torch.distributions.Categorical(GMM_WEIGHTS)
    k = cat.sample((n_samples,))
    x = torch.zeros((n_samples, 2), device=DEVICE)
    for i in range(len(GMM_WEIGHTS)):
        mask = (k == i)
        if mask.sum() > 0:
            mvn = torch.distributions.MultivariateNormal(GMM_MEANS[i], GMM_COVS[i])
            x[mask] = mvn.sample((mask.sum(),))
    return x

def compute_gmm_log_prob(pts):
    """Computes true GMM log-likelihood to measure manifold drift."""
    B = pts.shape[0]
    K = len(GMM_WEIGHTS)
    probs = torch.zeros((B, K), device=pts.device)
    for k in range(K):
        mvn = torch.distributions.MultivariateNormal(GMM_MEANS[k], GMM_COVS[k])
        probs[:, k] = GMM_WEIGHTS[k] * torch.exp(mvn.log_prob(pts))
    return torch.log(probs.sum(dim=-1) + 1e-12)


# Oracle ground truth with bayesian posterior parameters
def compute_conditioned_gmm_params():
    K = len(GMM_WEIGHTS)
    post_w = torch.zeros(K, device=DEVICE)
    post_mu = torch.zeros((K, 2), device=DEVICE)
    post_cov = torch.zeros((K, 2, 2), device=DEVICE)

    for k in range(K):
        mu_k = GMM_MEANS[k].unsqueeze(1)
        sigma_k = GMM_COVS[k]
        S = Q_MAT @ sigma_k @ Q_MAT.T + R_COV
        K_gain = sigma_k @ Q_MAT.T @ torch.inverse(S)
        
        mu_post = mu_k + K_gain @ (q_VAL.unsqueeze(1) - Q_MAT @ mu_k)
        sigma_post = sigma_k - K_gain @ Q_MAT @ sigma_k
        
        post_mu[k] = mu_post.squeeze(1)
        post_cov[k] = 0.5 * (sigma_post + sigma_post.T)
        dist_meas = torch.distributions.Normal((Q_MAT @ mu_k).squeeze(), torch.sqrt(S.squeeze()))
        post_w[k] = GMM_WEIGHTS[k] * torch.exp(dist_meas.log_prob(q_VAL.squeeze()))

    post_w = post_w / post_w.sum()
    return post_w, post_mu, post_cov

def compute_Dt_oracle(x, t, post_w, post_mu, post_cov):
    B = x.shape[0]
    K = len(post_w)
    t_safe = min(max(float(t), 1e-4), 0.9999)
    sigma_sq = (1.0 - t_safe) ** 2
    
    probs = torch.zeros((B, K), device=DEVICE)
    cond_means = torch.zeros((B, K, 2), device=DEVICE)
    I2 = torch.eye(2, device=DEVICE).unsqueeze(0)
    
    for k in range(K):
        mu_tk = t_safe * post_mu[k]
        Sigma_tk = sigma_sq * I2 + (t_safe ** 2) * post_cov[k].unsqueeze(0)
        mvn_t = torch.distributions.MultivariateNormal(mu_tk, Sigma_tk)
        probs[:, k] = post_w[k] * torch.exp(mvn_t.log_prob(x))
        
        cov_cross = t_safe * post_cov[k].unsqueeze(0)
        inv_sigma = torch.inverse(Sigma_tk)
        diff = (x - mu_tk).unsqueeze(-1)
        cond_means[:, k] = post_mu[k] + (cov_cross @ inv_sigma @ diff).squeeze(-1)
        
    gamma = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
    return torch.sum(gamma.unsqueeze(-1) * cond_means, dim=1)

# MeanFlow Architecture & Training

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class MeanFlowMLP(nn.Module):

    def __init__(self, data_dim=2, hidden_dim=256, time_dim=64):
        super().__init__()
        self.sin_embed = SinusoidalEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dt_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(data_dim, hidden_dim)
        self.b1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.b2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()

    def forward(self, z, t, r):
        # t: current time, r: destination time
        emb = self.time_mlp(self.sin_embed(t)) + self.dt_mlp(self.sin_embed(r - t))
        h = self.act(self.in_proj(z) + emb)
        h = h + self.b1(self.act(h + emb))
        h = h + self.b2(self.act(h + emb))
        return self.out_proj(h)


def train_meanflow(epochs=2500, batch_size=512, lr=5e-4):
    model = MeanFlowMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    params = dict(model.named_parameters())

    def forward_fn(z_in, t_in, r_in):
        return functional_call(model, params, (z_in, t_in, r_in))

    for step in range(epochs):
        x_1 = sample_gmm(batch_size)
        x_0 = torch.randn_like(x_1)

        # Sample t <= r in [0, 1]
        t1 = torch.rand(batch_size, 1, device=DEVICE)
        t2 = torch.rand(batch_size, 1, device=DEVICE)
        t = torch.minimum(t1, t2)  # Current time (closer to noise 0)
        r = torch.maximum(t1, t2)  # Target time (closer to data 1)

        # 25% anchor where r = t (instantaneous velocity field) paper does this
        mask = torch.rand(batch_size, 1, device=DEVICE) < 0.25
        r = torch.where(mask, t, r)

        # Linear interpolant: z_t at current time t
        z_t = (1.0 - t) * x_0 + t * x_1
        v_t = x_1 - x_0

        # JVP: derivative of u with respect to current time t
        primals = (z_t, t, r)
        tangents = (v_t, torch.ones_like(t), torch.zeros_like(r))
        u_pred, dudt = jvp(forward_fn, primals, tangents)

        # MeanFlow target: u_tgt = v_t - (r - t) * dudt
        u_tgt = (v_t + (r - t) * dudt).detach()

        diff_sq = torch.sum((u_pred - u_tgt) ** 2, dim=-1, keepdim=True)
        weight = 1.0 / torch.sqrt(diff_sq + 1e-3)
        loss = torch.mean(weight.detach() * diff_sq)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"Step {step+1:4d} | Loss: {loss.item():.4f}")

    return model


# Discrete Solvers across N Steps
def run_meanflow_oracle(init_pts, n_steps=8):
    post_w, post_mu, post_cov = compute_conditioned_gmm_params()
    dt = 1.0 / n_steps
    traj = [init_pts.clone().cpu().numpy()]
    x = init_pts.clone()

    for s in range(n_steps):
        t_val = s * dt
        t_safe = min(max(t_val, 1e-4), 0.9999)
        d_star = compute_Dt_oracle(x, t_safe, post_w, post_mu, post_cov)
        
        if n_steps == 1:
            x = d_star
        else:
            v_star = (d_star - x) / (1.0 - t_safe)
            x = x + dt * v_star
        traj.append(x.clone().cpu().numpy())
    return np.array(traj)

def run_meanflow_local_gaussian(init_pts, model, n_steps=8):
    """
    Row 1: Full-Covariance Local Gaussian on the trained MeanFlow model.
    Evaluates exact 2x2 Jacobian J_t and closed-form Kalman filter
    """
    model.eval()
    dt = 1.0 / n_steps
    traj = [init_pts.detach().cpu().numpy()]
    x = init_pts.clone()

    for s in range(n_steps):
        t_val = s * dt
        t_safe = min(max(t_val, 1e-4), 0.999)
        B = x.shape[0]

        t_tensor = torch.full((B, 1), t_safe, device=DEVICE)
        r_target = torch.full((B, 1), 1.0, device=DEVICE)

        # Clean endpoint prediction via MeanFlow chord
        x_req = x.detach().clone().requires_grad_(True)
        u_pred = model(x_req, t_tensor, r_target)
        d_t = x_req + (1.0 - t_safe) * u_pred

        # 2x2 Spatial Jacobian of the MeanFlow prediction
        j1 = torch.autograd.grad(d_t[:, 0].sum(), x_req, retain_graph=True)[0]
        j2 = torch.autograd.grad(d_t[:, 1].sum(), x_req)[0]
        J_t = torch.stack([j1, j2], dim=-1)  # (B, 2, 2)

        # Tweedie Covariance C_t = ((1 - t)^2 / t) * J_t
        prefactor = ((1.0 - t_safe) ** 2) / t_safe
        C_t = prefactor * J_t

        # Closed-form Kalman Conditioning
        res = q_VAL - (d_t @ Q_MAT.T)
        S = Q_MAT @ C_t @ Q_MAT.T + R_COV
        K_gain = (C_t @ Q_MAT.T) / S
        d_t_q = d_t + (K_gain @ res.unsqueeze(-1)).squeeze(-1)

        # Step
        if n_steps == 1:
            x = d_t_q
        else:
            v_corr = (d_t_q - x) / (1.0 - t_safe)
            with torch.no_grad():
                x = x + dt * v_corr

        traj.append(x.detach().cpu().numpy())

    return np.array(traj)

def run_meanflow_vanilla_dps(init_pts, model, n_steps=8, gamma=3.8):
    """
    Row 2: Vanilla MeanFlowDPS.
    Unregularized gradient guidance with 1/(1-t) velocity singularity.
    """
    model.eval()
    dt = 1.0 / n_steps
    traj = [init_pts.detach().cpu().numpy()]
    x = init_pts.clone()

    for s in range(n_steps):
        t_val = s * dt
        t_safe = min(max(t_val, 1e-4), 0.985)
        B = x.shape[0]

        t_tensor = torch.full((B, 1), t_safe, device=DEVICE)
        r_target = torch.full((B, 1), 1.0, device=DEVICE)

        with torch.no_grad():
            u_pred = model(x, t_tensor, r_target)
            d_t = x + (1.0 - t_safe) * u_pred

        res = q_VAL - (d_t @ Q_MAT.T)

        if n_steps == 1:
            # Single jump with fixed gradient step
            x = d_t + 0.65 * (res @ Q_MAT)
        else:
            v_prior = (d_t - x) / (1.0 - t_safe)
            v_guide = (gamma / (1.0 - t_safe)) * (res @ Q_MAT)
            v_total = v_prior + v_guide
            with torch.no_grad():
                x = x + dt * v_total

        traj.append(x.detach().cpu().numpy())

    return np.array(traj)







def psd_sym(J):
    e, V = torch.linalg.eigh(0.5 * (J + J.transpose(-1, -2)))
    return V @ torch.diag_embed(e.clamp_min(0.0)) @ V.transpose(-1, -2)

def cov_from_jacobian(J, rule="JJt"):
    if rule == "orth":
        return torch.eye(2, device=J.device).expand(J.shape[0], 2, 2)
    if rule == "symJ":
        return psd_sym(J)
    if rule == "JJt":
        return J @ J.transpose(-1, -2)
    raise ValueError(rule)

def condition(x, C, q=None, R=0.0, n=None):
    if n is None:
        n = Q_MAT.squeeze(0)
    if q is None:
        q = q_VAL.squeeze()
    Cn = torch.einsum("...ij,j->...i", C, n)
    if Cn.dim() == 1:
        Cn = Cn.expand_as(x)
    den = Cn @ n + R
    resid = q - x @ n
    scale = torch.linalg.vector_norm(Cn, dim=-1) * torch.linalg.vector_norm(n)
    bad = den <= 1e-6 * scale + 1e-12
    Cn = torch.where(bad[:, None], n.expand_as(x), Cn)
    den = torch.where(bad, n @ n + R, den)
    return x + Cn * (resid / den)[:, None]

def mf_jump(model, x, r, t):
    B = x.shape[0]
    r_t = torch.full((B, 1), float(r), device=x.device)
    t_t = torch.full((B, 1), float(t), device=x.device)
    return x + (t - r) * model(x, r_t, t_t)

def mf_jump_and_jac(model, x, r, t):
    x_req = x.detach().clone().requires_grad_(True)
    B = x.shape[0]
    r_t = torch.full((B, 1), float(r), device=x.device)
    t_t = torch.full((B, 1), float(t), device=x.device)
    y = x_req + (t - r) * model(x_req, r_t, t_t)
    j1 = torch.autograd.grad(y[:, 0].sum(), x_req, retain_graph=True)[0]
    j2 = torch.autograd.grad(y[:, 1].sum(), x_req)[0]
    J = torch.stack([j1, j2], dim=1)
    return y.detach(), J

_PRIOR_CACHE = {}

def prior_samples(model, n_prior=2000):
    if id(model) not in _PRIOR_CACHE:
        with torch.no_grad():
            torch.manual_seed(4242)
            z = torch.randn(n_prior, 2, device=DEVICE)
            _PRIOR_CACHE[id(model)] = mf_jump(model, z, 0.0, 1.0)
    return _PRIOR_CACHE[id(model)]

def is_moments(x, t, prior, chunk=1000):
    P2 = (prior[:, :, None] * prior[:, None, :]).reshape(-1, 4)
    Ds, Cs = [], []
    for xc in x.split(chunk):
        w = torch.softmax(-torch.cdist(xc, t * prior).square() / (2 * (1 - t) ** 2), dim=1)
        D = w @ prior
        Ds.append(D)
        Cs.append((w @ P2).reshape(-1, 2, 2) - D[:, :, None] * D[:, None, :])
    D, C = torch.cat(Ds), torch.cat(Cs)
    return D, 0.5 * (C + C.transpose(-1, -2))

T_IS = 0.3
T_MIN = 0.02

def lg_velocity_learned(model, x, t, early="IS"):
    if early == "IS" and t < T_IS:
        D, C = is_moments(x, t, prior_samples(model))
    else:
        te = max(float(t), T_MIN)
        B = x.shape[0]
        t_tensor = torch.full((B, 1), te, device=x.device)
        x_req = x.detach().clone().requires_grad_(True)
        u_inst = model(x_req, t_tensor, t_tensor)
        d_t = x_req + (1.0 - te) * u_inst
        j1 = torch.autograd.grad(d_t[:, 0].sum(), x_req, retain_graph=True)[0]
        j2 = torch.autograd.grad(d_t[:, 1].sum(), x_req)[0]
        J = torch.stack([j1, j2], dim=1)
        D = d_t.detach()
        C = ((1.0 - te) ** 2 / te) * psd_sym(J)
    return (condition(D, C) - x) / (1.0 - min(float(t), 0.9999))

def lg_flow(z, vel, steps=50):
    x, h, path = z.clone(), 1.0 / steps, [z.clone()]
    for i in range(steps):
        x = x + h * vel(x, i * h)
        path.append(x.clone())
    return x, torch.stack(path)

def hybrid(z, t_s, K, rule="JJt", model=None):
    path = [z.clone()]
    if K > 0:
        vel = lambda x_, t_: lg_velocity_learned(model, x_, t_)
        x, h = z.clone(), t_s / K
        for i in range(K):
            x = x + h * vel(x, i * h)
            path.append(x.clone())
        y, J = mf_jump_and_jac(model, x, t_s, 1.0)
    else:
        x = z if t_s == 0 else mf_jump(model, z, 0.0, t_s)
        y, J = mf_jump_and_jac(model, x, t_s, 1.0)
        if t_s > 0:
            path.append(x.clone())
    out = condition(y, cov_from_jacobian(J, rule))
    path.append(out.clone())
    return out, torch.stack(path)

def run_and_plot_flow_geometry(model):
    model.eval()
    z4 = generate_initial_grid()
    post_w, post_mu, post_cov = compute_conditioned_gmm_params()

    vel_oracle = lambda x_, t_: (compute_Dt_oracle(x_, min(max(float(t_), 1e-4), 0.9999), post_w, post_mu, post_cov) - x_) / (1.0 - min(max(float(t_), 1e-4), 0.9999))
    vel_lg = lambda x_, t_: lg_velocity_learned(model, x_, t_)

    paths = {
        "A. oracle conditional-training field": lg_flow(z4, vel_oracle, 50)[1].detach().cpu().numpy(),
        "B. LGC flow, learned MeanFlow (50 steps)": lg_flow(z4, vel_lg, 50)[1].detach().cpu().numpy(),
        "C. hybrid ts=0.5, K=5 (6 evaluations)": hybrid(z4, 0.5, 5, "JJt", model=model)[1].detach().cpu().numpy(),
        "D. one-step + JJᵀ (1 evaluation)": hybrid(z4, 0.0, 0, "JJt", model=model)[1].detach().cpu().numpy(),
    }

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharex=True, sharey=True)
    covs_np = GMM_COVS.cpu().numpy()
    means_np = GMM_MEANS.cpu().numpy()
    normal_np = Q_MAT.squeeze(0).cpu().numpy()
    q_val_float = float(q_VAL.item())

    for a, (title, P) in zip(axes, paths.items()):
        for k in range(3):
            e_, V_ = np.linalg.eigh(covs_np[k])
            ang = math.degrees(math.atan2(V_[1, 1], V_[0, 1]))
            for ns in (1, 2):
                a.add_patch(Ellipse(means_np[k].tolist(), 2 * ns * math.sqrt(e_[1]), 2 * ns * math.sqrt(e_[0]), angle=ang,
                                    fill=False, ec="#bbbbbb", lw=.8))
        xs = np.linspace(-4.5, 6.0, 2)
        a.plot(xs, (q_val_float - normal_np[0] * xs) / normal_np[1], "k--", lw=1)
        for j in range(P.shape[1]):
            a.plot(P[:, j, 0], P[:, j, 1], "-", color="tab:purple", alpha=.7, lw=.9)
            a.scatter(P[0, j, 0], P[0, j, 1], s=10, color="tab:blue", zorder=3)
            a.scatter(P[-1, j, 0], P[-1, j, 1], marker="x", s=25, color="tab:purple", zorder=3)
        a.set_title(title, fontsize=9)
        a.set_xlim(-4.5, 6.0)
        a.set_ylim(-5, 4)
        a.set_aspect("equal")

    fig.tight_layout()
    plt.savefig("fig_E4_flow_geometry.png", dpi=300, bbox_inches="tight")
    plt.savefig("fig_E4_flow_geometry.pdf", bbox_inches="tight")
    print("Saved 'fig_E4_flow_geometry.png' and 'fig_E4_flow_geometry.pdf'")


## visualization
def run_and_plot_meanflow_experiment(model, steps_list=[1, 2, 4, 8, 16, 32]):
    grid_init = generate_initial_grid()
    n_cols = len(steps_list)

    fig, axes = plt.subplots(
        3, n_cols, 
        figsize=(4.4 * n_cols, 12.8), 
        constrained_layout=True, 
        dpi=300
    )

    # 300x300 GMM background density grid
    gx = np.linspace(-3.5, 6.5, 300)
    gy = np.linspace(-5.0, 4.0, 300)
    GX, GY = np.meshgrid(gx, gy)
    pos = np.stack([GX, GY], axis=-1)
    
    pdf = np.zeros(GX.shape)
    for w, mu, sigma in zip(GMM_WEIGHTS.cpu().numpy(), GMM_MEANS.cpu().numpy(), GMM_COVS.cpu().numpy()):
        diff = pos - mu
        inv_sigma = np.linalg.inv(sigma)
        exponent = -0.5 * np.einsum('...k,kl,...l->...', diff, inv_sigma, diff)
        norm_const = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(sigma)))
        pdf += w * norm_const * np.exp(exponent)

    # Measurement line 0.625 * x1 + x2 = -1.5
    line_x = np.array([-4.5, 5.2])
    line_y = -0.625 * line_x - 1.5

    # 25 Distinct trajectory colors matching paper
    cmap = mpl.colormaps['tab20']
    colors = [cmap(i % 20) for i in range(25)]
    colors[20] = '#2ca02c'
    colors[21] = '#d62728'
    colors[22] = '#9467bd'
    colors[23] = '#8c564b'
    colors[24] = '#e377c2'

    means_np = GMM_MEANS.cpu().numpy()

    row_names = [
        "Oracle Flow (MeanFlow)",
        "Local Gaussian (Full Cov)",
        "Vanilla MeanFlowDPS"
    ]

    print("\nQUANTITATIVE STEP ABLATION")
    print(f"{'N Steps':<8} | {'Method':<26} | {'Measurement MSE':<16} | {'GMM LogP':<10}")

    for j, n_steps in enumerate(steps_list):
        traj_oracle = run_meanflow_oracle(grid_init, n_steps=n_steps)
        traj_lg = run_meanflow_local_gaussian(grid_init, model, n_steps=n_steps)
        traj_dps = run_meanflow_vanilla_dps(grid_init, model, n_steps=n_steps)

        methods_trajs = [traj_oracle, traj_lg, traj_dps]

        for i, traj in enumerate(methods_trajs):
            ax = axes[i, j]
            final_pts = torch.tensor(traj[-1], device=DEVICE)
            
            # Metrics
            mse = torch.mean((q_VAL - final_pts @ Q_MAT.T) ** 2).item()
            logp = torch.mean(compute_gmm_log_prob(final_pts)).item()

            print(f"{n_steps:<8d} | {row_names[i]:<26} | {mse:<16.4f} | {logp:<10.2f}")

            # Plot GMM Contours & Centroids
            ax.contourf(GX, GY, pdf, levels=14, cmap='Greys', alpha=0.10)
            ax.contour(GX, GY, pdf, levels=10, colors='gray', linewidths=0.5, alpha=0.40)
            ax.scatter(means_np[:, 0], means_np[:, 1], c='black', s=12, zorder=5)

            # Dashed Constraint Line
            ax.plot(line_x, line_y, color='black', linestyle='--', linewidth=1.1, dashes=(5, 3), zorder=3)

            # Plot Trajectories
            num_particles = traj.shape[1]
            for p in range(num_particles):
                c = colors[p % len(colors)]
                # Starting dot
                ax.scatter(traj[0, p, 0], traj[0, p, 1], color=c, s=24, zorder=6, edgecolors='none')
                # Trajectory chord path
                ax.plot(traj[:, p, 0], traj[:, p, 1], color=c, linewidth=1.2, alpha=0.85, zorder=4)
                # Ending cross
                ax.scatter(traj[-1, p, 0], traj[-1, p, 1], color=c, marker='x', s=38, linewidths=1.5, zorder=7)

            # Subplot titles
            if i == 0:
                col_header = f"N = {n_steps}\n"
            else:
                col_header = ""

            ax.set_title(f"{col_header}{row_names[i]}\nMSE: {mse:.3f} | LogP: {logp:.1f}", 
                         fontsize=10, loc='left', pad=4)
            ax.set_xlim(-3.5, 6.2)
            ax.set_ylim(-4.5, 3.5)
            ax.set_xticks([-2, 0, 2, 4, 6])
            ax.set_yticks([-4, -3, -2, -1, 0, 1, 2, 3])
            ax.tick_params(direction='out', length=3.5, width=0.8, labelsize=8)
            ax.set_aspect('equal')
            if j > 0:
                ax.tick_params(labelleft=False)


    plt.savefig("meanflow_ablation_N_hires.png", dpi=300, bbox_inches='tight')
    plt.savefig("meanflow_ablation_N_vector.pdf", format='pdf', bbox_inches='tight')
    print("\nSaved high-res PNG: 'meanflow_ablation_N_hires.png'")
    print("Saved vector PDF: 'meanflow_ablation_N_vector.pdf'")


if __name__ == '__main__':
    mf_model = train_meanflow(epochs=3000, batch_size=512, lr=6e-4)
    run_and_plot_flow_geometry(mf_model)
    run_and_plot_meanflow_experiment(mf_model, steps_list=[1, 2, 4, 8, 16, 32, 64, 128])