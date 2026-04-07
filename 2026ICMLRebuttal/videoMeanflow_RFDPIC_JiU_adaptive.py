import torch
from torchdiffeq import odeint

from videoMeanflow_RFDPIC_JiU import MeanFlow


class AdaptiveMeanFlow(MeanFlow):
    def __init__(self, *args, adaptive_method="dopri5", ode_rtol=1e-4, ode_atol=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_method = adaptive_method
        self.ode_rtol = float(ode_rtol)
        self.ode_atol = float(ode_atol)
        self.nfe_history = []

    @torch.no_grad()
    def sample_prediction(self, model, c_past_and_cond, sample_steps=5, device="cuda"):
        del sample_steps
        c_start, c_cond = c_past_and_cond
        model.eval()

        z0 = self.normer.norm(c_start)
        nfe = 0

        def rhs(t_scalar, z_state):
            nonlocal nfe
            nfe += 1

            t = torch.full((z_state.size(0),), float(t_scalar), device=z_state.device, dtype=z_state.dtype)
            r = t
            x_pred = model(z_state, t, r, c_cond)

            denom = t.clamp(min=1e-3).view(-1, 1, 1, 1, 1)
            v = (z_state - x_pred) / denom
            return v

        t_span = torch.tensor([1.0, 0.0], device=z0.device, dtype=z0.dtype)
        z_traj = odeint(
            rhs,
            z0,
            t_span,
            method=self.adaptive_method,
            rtol=self.ode_rtol,
            atol=self.ode_atol,
        )
        self.nfe_history.append(int(nfe))
        return self.normer.unnorm(z_traj[-1])
