import numpy as np
import torch.distributions

from guided_diffusion.gaussian_diffusion import ModelVarType, ModelMeanType  # get_named_beta_schedule, betas_for_alpha_bar
from guided_diffusion.gaussian_diffusion import _extract_into_tensor
from .base_diffusion import BaseDiffusion
from .base_diffusion import get_model_mean_type, get_model_var_type, get_loss_type

class ColdMixDiffusion(BaseDiffusion):
    def __init_originalNU__(self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timesteps=False):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.timesteps_scaling_factor = 1000.0

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, x_T_end=None):
        """
        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param x_T_end: if specified, the end point x_T (noise / noisy image).
        :return: A mixed version of x_start.
        """
        assert x_T_end is not None
        assert x_T_end.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_T_end
        )
    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, x_T_end=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            raise NotImplementedError
        else:
            if self.model_var_type == ModelVarType.FIXED_LARGE:
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                model_variance, model_log_variance = np.append(self.posterior_variance[1], self.betas[1:]),\
                                                     np.log(np.append(self.posterior_variance[1], self.betas[1:]))
            elif self.model_var_type == ModelVarType.FIXED_SMALL:
                model_variance, model_log_variance = self.posterior_variance,\
                                                     self.posterior_log_variance_clipped
            else:
                raise ValueError(self.model_var_type)

            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def sample_strategy(self, pred_xstart, xt, t, x_T_end):
        '''
        Sample x_{t-1} from x_t , x_start and x_T. Choose way to sampls: naive /cold.
        @param pred_xstart:
        @param xt:
        @param t:
        @param x_T_end:
        @return:
        '''
        if torch.any(t == 0):
            if torch.all(t == 0):
                Warning('stop sampling stategy since some t in batch reached 0')
            return pred_xstart
        #basic = self.q_sample(pred_xstart, t-1, x_T_end) # maybe unstable
        coldsample = xt - self.q_sample(pred_xstart, t, x_T_end) + self.q_sample(pred_xstart, t-1, x_T_end)
        return coldsample
    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None,x_T_end=None):
        """
        Sample x_{t-1} from the model at the given timestep x_t.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        assert x_T_end is not None
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,)
        #noise = torch.randn_like(x)
        #nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        #sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        sample = self.sample_strategy(pred_xstart=out["pred_xstart"], xt=x, t=t, x_T_end=x_T_end)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None):
        if 'x_T_end' not in model_kwargs:
            raise ValueError('Should be for coldmix diffusion!')
        return super().training_losses(model, x_start, t, model_kwargs)

class NoiseDiffusion(BaseDiffusion):
    def __init__(self, betas, model_mean_type_name, model_var_type_name, loss_type_name,
                 rescale_timesteps=False, **kwargs):
        self.model_mean_type = get_model_mean_type(model_mean_type_name)
        self.model_var_type = get_model_var_type(model_var_type_name)
        self.loss_type = get_loss_type(loss_type_name)
        self.rescale_timesteps = rescale_timesteps
        self.timesteps_scaling_factor = 1000.0

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        '''
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        '''
        #self.log_Lr = torch.linspace(np.log10(0.0001)*2, np.log10(0.012), self.num_timesteps)
        self.log_Ls = torch.log10(torch.linspace(0.00000001, 0.012, self.num_timesteps)) # todo check sched.
        self.log_Lr = 2.18 * self.log_Ls + 1.2
        lambdas_std = 0.4 # in paper: 0.26
        self.log_var_L = torch.linspace(0.001, lambdas_std**2, self.num_timesteps)/1 # todo reset variance

    def _get_log_mean_Lr_Ls(self, t):
        log_mean_Lr = self.log_Lr.to(device=t.device)[t].float()
        log_mean_Ls = self.log_Ls.to(device=t.device)[t].float()
        return log_mean_Lr, log_mean_Ls
    def q_sample(self, x_start, t, x_T_end=None, time_T_end=None):
        """
        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param x_T_end: if specified, the end point x_T (noise / noisy image).
        :param time_T_end: in vase T~ is not T. i.e starting at T!=1000 (noise level, etc)
        :return: A mixed version of x_start.
        """
        if x_T_end is None:
            return self.q_sample_train(x_start, t)
        #assert xt is not None
        assert x_T_end.shape == x_start.shape
        #norm_scale_img01 = lambda x: torch.clip((x + 1) / 2, 0, 1)

        if time_T_end is None:
            time_T_end = self.num_timesteps - 1
        log_mean_Lr_end, log_mean_Ls_end = self._get_log_mean_Lr_Ls(torch.ones_like(t) * time_T_end)
        Lr = NoiseDiffusion._shape_broadcast(10 ** log_mean_Lr_end, x_start.shape)
        Ls = NoiseDiffusion._shape_broadcast(10 ** log_mean_Ls_end, x_start.shape)
        #todo: if take these or t_init values or metadata values
        variance_end = Lr + NoiseDiffusion.norm_scale_to01(x_start) * Ls

        log_mean_Lr_t, log_mean_Ls_t = self._get_log_mean_Lr_Ls(t)
        #print(log_mean_Lr_end, log_mean_Lr_t)
        #print(log_mean_Ls_end, log_mean_Ls_t)
        assert torch.all(log_mean_Lr_end>=log_mean_Lr_t) and torch.all(log_mean_Ls_end>=log_mean_Ls_t), \
            f'at t: {t} - {log_mean_Lr_t, log_mean_Ls_t}; at T: {time_T_end} - {log_mean_Lr_end, log_mean_Ls_end}'

        Lr = NoiseDiffusion._shape_broadcast(10**log_mean_Lr_t, x_start.shape)
        Ls = NoiseDiffusion._shape_broadcast(10**log_mean_Ls_t, x_start.shape)
        variance_t = Lr + NoiseDiffusion.norm_scale_to01(x_start) * Ls
        # took root and norm N_est by sqrt(var_t/var_end) >> N_T changed its variance to N_t variance
        N_T_estimated = x_T_end - x_start
        N_t = torch.sqrt(variance_t/variance_end) * N_T_estimated
        # > add to x_start
        x_t = x_start + N_t # x at t_minus1 using variance normalization noise Nt
        return x_t
    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, x_T_end=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE] and False: # todo remove
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        elif False:
            if self.model_var_type == ModelVarType.FIXED_LARGE:
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                model_variance, model_log_variance = np.append(self.posterior_variance[1], self.betas[1:]),\
                                                     np.log(np.append(self.posterior_variance[1], self.betas[1:]))
            elif self.model_var_type == ModelVarType.FIXED_SMALL:
                model_variance, model_log_variance = self.posterior_variance,\
                                                     self.posterior_log_variance_clipped
            else:
                raise ValueError(self.model_var_type)

            # NU model_variance = _extract_into_tensor(model_variance, t, x.shape)
            #model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
        model_variance, model_log_variance = 0,0

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = None # process_xstart(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            # NU model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
            model_mean = None
        else:
            raise NotImplementedError(self.model_mean_type)

        #assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape) todo comment in
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    def _predict_xstart_from_eps(self, x_t, t, eps):
        # Reconstruct X0 from Xt with Nt_pred > Xt = X0+Nt > X0 = Xt-Nt
        assert x_t.shape == eps.shape
        return x_t - eps
    def sample_strategy(self, pred_xstart, xt, t, x_T_end, time_T_end):
        '''
        Sample x_{t-1} from x_t , x_start and x_T. Choose way to sampls: naive /cold.
        @param pred_xstart: x0 - clean image
        @param xt:
        @param t:
        @param x_T_end: noisy image
        @return:
        '''
        if torch.any(t == 0):
            if not torch.all(t == 0):
                Warning('stop sampling stategy since some t in batch reached 0 BUT NOT ALL')
            return pred_xstart
        #basic = self.q_sample(pred_xstart, t-1, x_T_end) # maybe unstable
        coldsample = xt - self.q_sample(pred_xstart, t, x_T_end, time_T_end) + self.q_sample(pred_xstart, t-1, x_T_end, time_T_end)
        return coldsample

    @torch.no_grad()
    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None,
                       model_kwargs=None, device=None, progress=False, diffusion_start_point=-1, x_start=None, get_x_T=False):
        final = None
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        #end_T_point = self.num_timesteps # start from time T
        Lr = model_kwargs.get('noise_Lr') if model_kwargs is not None else None
        Ls = model_kwargs.get('noise_Ls') if model_kwargs is not None else None
        end_T_point = self.find_denoiser_start_point(Lr, Ls)
        end_T = torch.tensor([end_T_point] * shape[0], device=device) # the final time index

        if noise is not None:  # for (real world) given noisy image
            img = noise
        elif x_start is not None:  # use (sample) simulative image instead
            img = self.q_sample(x_start, end_T)  # noise #change to sidd
        else:
            img = torch.randn(*shape, device=device)
        # sample the input image at time T:

        x_T_end = img
        time_T_end = end_T_point
        '''
        if diffusion_start_point != -1:
            end_T_point = diffusion_start_point # start from the specified time "t1"
            time_vec = torch.tensor([end_T_point] * shape[0], device=device)
            img = self.q_sample(model_kwargs['img2'], time_vec) # init img to start point sample: X_t1
            print('start sampling from t_step: ', end_T_point)
        '''
        indices = list(range(end_T_point))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            out = self.p_sample(
                model,
                img,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                x_T_end=x_T_end,
                time_T_end=time_T_end
            )
            #yield out
            img = out["sample"]

        if get_x_T:
            return img, x_T_end
        return img

    def find_denoiser_start_point(self, Lr=None, Ls=None):
        if Lr is None or Ls is None:
            #print('Lr/Ls is missing')
            return self.num_timesteps - 1
        # print('NOT IMPLEMENTED')
        def print(*k):
            pass
        print(Lr.shape, Ls.shape)
        log_Lr = torch.log10(Lr).unsqueeze(1) # shape: B,1
        log_Ls = torch.log10(Ls).unsqueeze(1)
        print('lr', log_Lr)
        print('ls', log_Ls)
        cur_device = log_Lr.device
        pr = (log_Lr - self.log_Lr.to(cur_device))**2 # shape: B, 1000
        ps = (log_Ls - self.log_Ls.to(cur_device))**2
        print('pr shape', pr.shape)
        probs = -1/(2*self.log_var_L.to(cur_device)) * (pr + ps) # B,1000
        print('probs', probs.shape)
        probs = -torch.log(2*np.pi*self.log_var_L.to(cur_device)) + probs
        # probs in shape Batchsize,T
        start_point = torch.argmax(probs, dim=1)
        print(start_point)
        return start_point.item()
        #return self.num_timesteps

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None,
                 x_T_end=None, time_T_end=None):
        """
        Sample x_{t-1} from the model at the given timestep x_t.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        assert x_T_end is not None
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,)
        #noise = torch.randn_like(x)
        #nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        if cond_fn is not None:
            #out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
            pass
        #sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        if out['mean'] is not None: # ie ModelMeanType.PREVIOUS_X
            return {"sample": out['mean'], "pred_xstart": out["pred_xstart"]}
        sample = self.sample_strategy(pred_xstart=out["pred_xstart"], xt=x, t=t, x_T_end=x_T_end, time_T_end=time_T_end)
        #
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None):
        model_kwargs['x_t'] = self.q_sample(x_start, t)
        model_kwargs.pop('x_T_end', None) # NU, so just in case (to prevent problems/bugs).
        return super().training_losses(model, x_start, t, model_kwargs)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        t_mask = t==0
        t[t==0] = 1 # prevent a bug in q_sample
        posterior_mean = self.q_sample(x_start, t - 1, x_T_end=x_t, time_T_end=t)
        posterior_mean[t_mask, ...] = x_start[t_mask, ...]
        posterior_variance = None
        posterior_log_variance_clipped = None
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample_train(self, x_start, t):
        '''
        In other words, sample from q(x_t | x_0).

        Sample noise using randomization of Lambda_read and Lambda_shot:
        sample L_r from P(L_r|t) from loglinear
        sample L_s from P(L_s|t) from loglinear
        compute variance: L_r +x*L_s
        sample gaussian noise N from P(N|Ls,Lr)
        @param x_start:
        @param t:
        @return:
        '''

        log_mean_Lr, log_mean_Ls = self._get_log_mean_Lr_Ls(t)
        # log_mean_Ls = self.log_Ls.to(device=t.device)[t].float()
        log_std = torch.sqrt(self.log_var_L.to(device=t.device)[t].float())
        #Ls = torch.distributions.log_normal.LogNormal(log_mean_Ls, log_std).sample()
        #Lr = torch.distributions.log_normal.LogNormal(log_mean_Lr, log_std).sample()
        Lr = 10 ** torch.distributions.Normal(log_mean_Lr, log_std).sample()
        Ls = 10 ** torch.distributions.Normal(log_mean_Ls, log_std).sample()
        #print(Lr, Ls)
        Lr = NoiseDiffusion._shape_broadcast(Lr, x_start.shape)
        Ls = NoiseDiffusion._shape_broadcast(Ls, x_start.shape)
        #norm_scale_img01 = lambda x: torch.clip( (x+1)/2, 0, 1)
        #norm_scale_img11 = lambda x: torch.clip(x*2-1, -1, 1)
        #print(Lr.shape)
        #print(Lr.shape, Ls.shape, x_start.shape)
        variance = Lr + NoiseDiffusion.norm_scale_to01(x_start)*Ls
        noise = torch.distributions.normal.Normal(0, torch.sqrt(variance)).sample()
        #print(x_start.mean(), x_start.min(), x_start.max())
        return x_start + noise*2#NoiseDiffusion.norm_scale_to11(noise) change scale according to dynamic range

    @staticmethod
    def norm_scale_to11(x, clip=False):
        '''

        @param x: tensor in dynamic range (0, 1)
        @return: tensor in dynamic range (-1, 1)
        '''
        if clip:
            return torch.clip(x*2-1, -1, 1)
        else:
            return x*2-1
    @staticmethod
    def norm_scale_to01(x, clip=True):
        '''

        @param x: tensor in dynamic range (-1, 1)
        @return: tensor in dynamic range (0, 1)
        '''
        if clip:
            return torch.clip( (x+1)/2, 0, 1)
        else:
            return (x+1)/2


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

import math
import torch.nn as nn
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# model

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

import torch


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        train_routine = 'Final',
        sampling_routine='default',
        discrete=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        self.train_routine = train_routine
        self.sampling_routine = sampling_routine

    @torch.no_grad()
    def sample(self, batch_size = 16, img=None, t=None):

        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps

        xt = img
        direct_recons = None

        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x1_bar = self.denoise_fn(img, step)
            x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)

            if direct_recons is None:
                direct_recons = x1_bar

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long).cuda()
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        self.denoise_fn.train()

        return xt, direct_recons, img

    def get_x2_bar_from_xt(self, x1_bar, xt, t):
        return (
                (xt - extract(self.sqrt_alphas_cumprod, t, x1_bar.shape) * x1_bar) /
                extract(self.sqrt_one_minus_alphas_cumprod, t, x1_bar.shape)
        )

    @torch.no_grad()
    def gen_sample(self, batch_size=16, img=None, noise_level=0, t=None):
        self.denoise_fn.eval()
        if t == None:
            t = self.num_timesteps

        noise = img
        direct_recons = None
        img = img + torch.randn_like(img) * noise_level

        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            x1_bar = self.denoise_fn(img, step)
            x2_bar = noise
            if direct_recons == None:
                direct_recons = x1_bar

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return noise, direct_recons, img

    @torch.no_grad()
    def forward_and_backward(self, batch_size=16, img1=None, img2=None, t=None, times=None, eval=True):

        self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps


        img = img1

        Forward = []
        Forward.append(img)

        noise = img2

        for i in range(t):
            with torch.no_grad():
                step = torch.full((batch_size,), i, dtype=torch.long, device=img.device)
                n_img = self.q_sample(x_start=img, x_end=noise, t=step)
                Forward.append(n_img)

        Backward = []
        img = n_img
        while (t):
            step = torch.full((batch_size,), t - 1, dtype=torch.long, device=img.device)
            x1_bar = self.denoise_fn(img, step)
            x2_bar = noise #self.get_x2_bar_from_xt(x1_bar, img, step)

            Backward.append(img)

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long, device=img.device)
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return Forward, Backward, img

    @torch.no_grad()
    def all_sample(self, batch_size=16, img=None, t=None, times=None, eval=True):

        if eval:
            self.denoise_fn.eval()

        if t == None:
            t = self.num_timesteps

        X1_0s, X2_0s, X_ts = [], [], []
        while (t):

            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x1_bar = self.denoise_fn(img, step)
            x2_bar = self.get_x2_bar_from_xt(x1_bar, img, step)


            X1_0s.append(x1_bar.detach().cpu())
            X2_0s.append(x2_bar.detach().cpu())
            X_ts.append(img.detach().cpu())

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long).cuda()
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img - xt_bar + xt_sub1_bar
            img = x
            t = t - 1

        return X1_0s, X_ts

    def q_sample(self, x_start, x_end, t):
        # simply use the alphas to interpolate
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_end
        )

    def p_losses(self, x_start, x_end, t):
        b, c, h, w = x_start.shape
        if self.train_routine == 'Final':
            x_mix = self.q_sample(x_start=x_start, x_end=x_end, t=t)
            x_recon = self.denoise_fn(x_mix, t)
            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon)
            else:
                raise NotImplementedError()

        return loss

    def forward(self, x1, x2, *args, **kwargs):
        b, c, h, w, device, img_size, = *x1.shape, x1.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x1, x2, t, *args, **kwargs)
import torch.utils.data
class Dataset_Aug1(torch.utils.data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)
