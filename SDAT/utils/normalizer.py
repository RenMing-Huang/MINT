import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch


@dataclass
class NormalizerState:
	norm_type: str  # 'Gaussian' | 'MinMax' | 'Identity'
	action_dim: int
	params: Dict[str, Any]

	def to_jsonable(self) -> Dict[str, Any]:
		def to_list(x):
			if x is None:
				return None
			if isinstance(x, (list, tuple)):
				return list(x)
			if isinstance(x, (np.ndarray,)):
				return x.tolist()
			if torch.is_tensor(x):
				return x.detach().cpu().numpy().tolist()
			return x

		return {
			"type": self.norm_type,
			"action_dim": int(self.action_dim),
			"params": {k: to_list(v) for k, v in self.params.items()},
		}

	@staticmethod
	def from_dict(d: Dict[str, Any]) -> "NormalizerState":
		return NormalizerState(
			norm_type=d["type"],
			action_dim=int(d["action_dim"]),
			params=d.get("params", {}),
		)


class BaseNormalizer:
	def __init__(self, action_dim: int, eps: float = 1e-6):
		self.action_dim = int(action_dim)
		self.eps = float(eps)

	def normalize(self, x: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError

	def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
		raise NotImplementedError

	def state(self) -> NormalizerState:
		raise NotImplementedError

	@staticmethod
	def load(path: str) -> "BaseNormalizer":
		with open(path, "r") as f:
			d = json.load(f)
		st = NormalizerState.from_dict(d)
		typ = st.norm_type.lower()
		if typ == "gaussian":
			mean = torch.tensor(st.params.get("mean"), dtype=torch.float32)
			std = torch.tensor(st.params.get("std"), dtype=torch.float32)
			eps = float(st.params.get("eps", 1e-6))
			return GaussianNormalizer(action_dim=st.action_dim, mean=mean, std=std, eps=eps)
		elif typ == "minmax":
			min_v = torch.tensor(st.params.get("min"), dtype=torch.float32)
			max_v = torch.tensor(st.params.get("max"), dtype=torch.float32)
			# Support quantile-based normalization (if saved with q01/q99)
			q01 = st.params.get("q01")
			q99 = st.params.get("q99")
			if q01 is not None:
				q01 = torch.tensor(q01, dtype=torch.float32)
			if q99 is not None:
				q99 = torch.tensor(q99, dtype=torch.float32)
			eps = float(st.params.get("eps", 1e-6))
			return MinMaxNormalizer(action_dim=st.action_dim, min_v=min_v, max_v=max_v, q01=q01, q99=q99, eps=eps)
		elif typ == "cdf":
			probs = torch.tensor(st.params.get("probs"), dtype=torch.float32)
			values = torch.tensor(st.params.get("values"), dtype=torch.float32)
			eps = float(st.params.get("eps", 1e-6))
			return CDFNormalizer(action_dim=st.action_dim, probs=probs, values=values, eps=eps)
		elif typ == "identity":
			return IdentityNormalizer(action_dim=st.action_dim)
		else:
			raise ValueError(f"Unknown normalizer type: {st.norm_type}")

	def save(self, path: str):
		with open(path, "w") as f:
			json.dump(self.state().to_jsonable(), f, indent=2)


class GaussianNormalizer(BaseNormalizer):
	def __init__(self, action_dim: int, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None, eps: float = 1e-6):
		super().__init__(action_dim, eps)
		if mean is None:
			mean = torch.zeros(action_dim, dtype=torch.float32)
		if std is None:
			std = torch.ones(action_dim, dtype=torch.float32)
		self.register_params(mean, std)

	def register_params(self, mean: torch.Tensor, std: torch.Tensor):
		mean = mean.reshape(-1)
		std = std.reshape(-1)
		assert mean.shape[0] == self.action_dim and std.shape[0] == self.action_dim
		self.mean = mean.detach().clone().float()
		self.std = std.detach().clone().float()

	def normalize(self, x: torch.Tensor) -> torch.Tensor:
		params_dev = self.mean.to(device=x.device), self.std.to(device=x.device)
		mean, std = params_dev
		return (x - mean) / (std + self.eps)

	def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
		params_dev = self.mean.to(device=x.device), self.std.to(device=x.device)
		mean, std = params_dev
		return x * (std + self.eps) + mean

	def state(self) -> NormalizerState:
		return NormalizerState(norm_type="Gaussian", action_dim=self.action_dim, params={
			"mean": self.mean,
			"std": self.std,
			"eps": self.eps,
		})


class MinMaxNormalizer(BaseNormalizer):
	def __init__(self, action_dim: int, min_v: Optional[torch.Tensor] = None, max_v: Optional[torch.Tensor] = None, 
	             q01: Optional[torch.Tensor] = None, q99: Optional[torch.Tensor] = None, eps: float = 1e-6):
		"""
		MinMax normalizer that scales actions to [-1, 1].
		
		Args:
			action_dim: Action dimension
			min_v: Absolute minimum values (fallback if quantiles not provided)
			max_v: Absolute maximum values (fallback if quantiles not provided)
			q01: 1st percentile (recommended for outlier-robust normalization)
			q99: 99th percentile (recommended for outlier-robust normalization)
			eps: Small value to avoid division by zero
		"""
		super().__init__(action_dim, eps)
		
		# Prefer quantiles over absolute min/max
		if q01 is not None and q99 is not None:
			self.use_quantiles = True
			low = q01
			high = q99
		else:
			self.use_quantiles = False
			low = min_v if min_v is not None else torch.zeros(action_dim, dtype=torch.float32)
			high = max_v if max_v is not None else torch.ones(action_dim, dtype=torch.float32)
		
		self.register_params(low, high)

	def register_params(self, min_v: torch.Tensor, max_v: torch.Tensor):
		min_v = min_v.reshape(-1)
		max_v = max_v.reshape(-1)
		assert min_v.shape[0] == self.action_dim and max_v.shape[0] == self.action_dim
		self.min_v = min_v.detach().clone().float()
		self.max_v = max_v.detach().clone().float()

	def normalize(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Normalize to [-1, 1].
		If using quantiles, clips outliers to [q01, q99] before scaling.
		"""
		min_v = self.min_v.to(x.device)
		max_v = self.max_v.to(x.device)
		
		# Clip outliers if using quantile-based normalization
		# if self.use_quantiles:
		# 	x = torch.clamp(x, min_v, max_v)
		
		scale = (max_v - min_v).clamp(min=self.eps)  # Avoid division by zero for constant dims
		x01 = (x - min_v) / scale
		return x01 * 2.0 - 1.0

	def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Unnormalize from [-1, 1] back to original scale.
		No clipping during unnormalization (allow reconstruction to exceed bounds).
		"""
		min_v = self.min_v.to(x.device)
		max_v = self.max_v.to(x.device)
		scale = (max_v - min_v).clamp(min=self.eps)
		x01 = (x + 1.0) * 0.5
		return x01 * scale + min_v

	def state(self) -> NormalizerState:
		params = {
			"min": self.min_v,
			"max": self.max_v,
			"eps": self.eps,
			"use_quantiles": self.use_quantiles,
		}
		# Save q01/q99 if using quantile-based normalization
		if self.use_quantiles:
			params["q01"] = self.min_v  # min_v stores q01 when using quantiles
			params["q99"] = self.max_v  # max_v stores q99 when using quantiles
		return NormalizerState(norm_type="MinMax", action_dim=self.action_dim, params=params)


class IdentityNormalizer(BaseNormalizer):
	def __init__(self, action_dim: int):
		super().__init__(action_dim)

	def normalize(self, x: torch.Tensor) -> torch.Tensor:
		return x

	def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
		return x

	def state(self) -> NormalizerState:
		return NormalizerState(norm_type="Identity", action_dim=self.action_dim, params={"eps": self.eps})


class CDFNormalizer(BaseNormalizer):
	"""
	Per-dimension CDF normalizer using piecewise-linear empirical CDF.
	- Normalize: y = 2 * CDF(x) - 1 in [-1, 1]
	- Unnormalize: x = CDF^{-1}((y + 1) / 2)
	Stores fixed knots:
	  probs: (K,) monotonically increasing in [0,1]
	  values: (D, K) sorted per-dimension values at those probs
	"""
	def __init__(self, action_dim: int, probs: torch.Tensor, values: torch.Tensor, eps: float = 1e-6):
		super().__init__(action_dim, eps)
		assert probs.dim() == 1, "probs must be 1-D of shape (K,)"
		assert values.dim() == 2 and values.shape[0] == action_dim, "values must be (D, K)"
		K = probs.shape[0]
		assert values.shape[1] == K, "values second dim must equal probs length"
		self.probs = probs.detach().clone().float().clamp(0.0, 1.0)
		self.values = values.detach().clone().float()

	def normalize(self, x: torch.Tensor) -> torch.Tensor:
		orig_shape = x.shape
		x2d = x.reshape(-1, self.action_dim)
		device = x2d.device
		vals = self.values.to(device)
		probs = self.probs.to(device)
		K = probs.numel()
		out = torch.empty_like(x2d)
		for d in range(self.action_dim):
			vd = vals[d]
			idx = torch.searchsorted(vd, x2d[:, d], right=False)
			idx = idx.clamp(min=1, max=K-1)
			x0 = vd[idx - 1]
			x1 = vd[idx]
			p0 = probs[idx - 1]
			p1 = probs[idx]
			denom = (x1 - x0).abs().clamp_min(self.eps)
			t = (x2d[:, d] - x0) / denom
			p = p0 + t * (p1 - p0)
			out[:, d] = p.mul(2.0).sub(1.0)
		return out.reshape(orig_shape)

	def unnormalize(self, y: torch.Tensor) -> torch.Tensor:
		orig_shape = y.shape
		y2d = y.reshape(-1, self.action_dim)
		device = y2d.device
		vals = self.values.to(device)
		probs = self.probs.to(device)
		K = probs.numel()
		out = torch.empty_like(y2d)
		p_in = (y2d + 1.0) * 0.5
		p_in = p_in.clamp(0.0, 1.0)
		for d in range(self.action_dim):
			vd = vals[d]
			idx = torch.searchsorted(probs, p_in[:, d], right=False)
			idx = idx.clamp(min=1, max=K-1)
			p0 = probs[idx - 1]
			p1 = probs[idx]
			v0 = vd[idx - 1]
			v1 = vd[idx]
			denom = (p1 - p0).abs().clamp_min(self.eps)
			t = (p_in[:, d] - p0) / denom
			out[:, d] = v0 + t * (v1 - v0)
		return out.reshape(orig_shape)

	def state(self) -> NormalizerState:
		return NormalizerState(norm_type="CDF", action_dim=self.action_dim, params={
			"probs": self.probs,
			"values": self.values,
			"eps": self.eps,
		})


@torch.no_grad()
def compute_action_stats_from_loader(loader, device: Optional[torch.device] = None, max_samples: int = 500000):
	"""
	Compute per-dimension mean, std, min, max, and quantiles over actions in a DataLoader.
	Assumes actions are under key 'actions' in each batch and the last dim is action_dim.
	
	Args:
		loader: DataLoader with actions
		device: Device to use for computation
		max_samples: Maximum number of action samples to collect for quantile computation
	
	Returns: 
		dict(mean, std, min, max, q01, q99, count, action_dim)
	"""
	count = 0
	mean = None
	m2 = None
	running_min = None
	running_max = None
	action_dim = None
	all_actions = []  # Collect for quantile computation

	for batch in loader:
		if isinstance(batch, dict):
			actions = batch.get("actions")
		else:
			actions = batch

		if device is not None:
			actions = actions.to(device)
		actions = actions.detach().to("cpu", dtype=torch.float32)
		a = actions[:, 0]
		if a.ndim == 1:
			a = a.unsqueeze(0)
		# flatten all but last dim
		a = a.reshape(-1, a.shape[-1])
		if action_dim is None:
			action_dim = int(a.shape[-1])
		
		# Collect for quantiles (with reservoir sampling if needed)
		if len(all_actions) < max_samples:
			all_actions.append(a)
		elif count < max_samples:
			# Already at capacity, use reservoir sampling
			for i in range(a.shape[0]):
				j = torch.randint(0, count + i + 1, (1,)).item()
				if j < max_samples:
					# Find which tensor and index to replace
					idx = j
					for tensor_idx, tensor in enumerate(all_actions):
						if idx < tensor.shape[0]:
							all_actions[tensor_idx][idx] = a[i]
							break
						idx -= tensor.shape[0]
		
		# per-batch stats for mean/std
		batch_n = a.shape[0]
		batch_mean = a.mean(dim=0)
		batch_var = a.var(dim=0, unbiased=False)
		batch_min = a.min(dim=0).values
		batch_max = a.max(dim=0).values

		if mean is None:
			mean = batch_mean
			m2 = batch_var * batch_n
			running_min = batch_min
			running_max = batch_max
			count = batch_n
		else:
			delta = batch_mean - mean
			total = count + batch_n
			new_mean = mean + delta * (batch_n / total)
			# update M2 (sum of squares of differences from the current mean)
			m2 = m2 + batch_var * batch_n + (delta ** 2) * (count * batch_n / total)
			mean = new_mean
			count = total
			running_min = torch.minimum(running_min, batch_min)
			running_max = torch.maximum(running_max, batch_max)

	if count == 0 or mean is None:
		raise RuntimeError("No actions found in loader to compute stats.")

	var = m2 / max(count, 1)
	std = torch.sqrt(var + 1e-12)
	
	# Compute quantiles from collected samples
	all_actions_concat = torch.cat(all_actions, dim=0)  # [N, D]
	q01 = torch.quantile(all_actions_concat, 0.01, dim=0)  # [D]
	q99 = torch.quantile(all_actions_concat, 0.99, dim=0)  # [D]
	
	return {
		"mean": mean,
		"std": std,
		"min": running_min,
		"max": running_max,
		"q01": q01,
		"q99": q99,
		"count": count,
		"action_dim": action_dim,
	}


@torch.no_grad()
def compute_action_cdf_knots_from_loader(loader, max_samples: int = 200000, num_bins: int = 1001, device: Optional[torch.device] = None):
	"""
	Estimate per-dimension empirical CDF with fixed quantile knots.
	Uses reservoir sampling up to max_samples for memory efficiency.
	Returns dict with probs (K,), values (D,K), action_dim, count_sampled.
	"""
	assert num_bins >= 3 and num_bins % 2 == 1, "num_bins should be odd and >=3 for symmetry"
	probs = torch.linspace(0.0, 1.0, steps=num_bins)
	samples = None
	total = 0
	for batch in loader:
		actions = batch.get("actions") if isinstance(batch, dict) else batch
		if actions is None:
			continue
		a = actions
		if device is not None:
			a = a.to(device)
		a = a.detach().to("cpu", dtype=torch.float32)
		if a.ndim == 1:
			a = a.unsqueeze(0)
		a = a.reshape(-1, a.shape[-1])
		if samples is None:
			if a.shape[0] > max_samples:
				idx = torch.randperm(a.shape[0])[:max_samples]
				samples = a[idx]
			else:
				samples = a.clone()
		else:
			need = max(0, max_samples - samples.shape[0])
			if need > 0:
				take = min(need, a.shape[0])
				if take > 0:
					idx = torch.randperm(a.shape[0])[:take]
					samples = torch.cat([samples, a[idx]], dim=0)
		total += a.shape[0]

	if samples is None:
		raise RuntimeError("No actions found to compute CDF knots.")

	q_KD = torch.quantile(samples, probs, dim=0)  # (K, D)
	values = q_KD.transpose(0, 1).contiguous()    # (D, K)
	return {
		"probs": probs,
		"values": values,
		"action_dim": int(samples.shape[1]),
		"count": int(total),
	}


def build_normalizer_from_stats(norm_type: str, stats: Dict[str, Any], eps: float = 1e-6) -> BaseNormalizer:
	t = norm_type.lower()
	ad = int(stats["action_dim"])
	if t == "gaussian":
		return GaussianNormalizer(action_dim=ad, mean=stats["mean"], std=stats["std"], eps=eps)
	elif t == "minmax":

		return MinMaxNormalizer(action_dim=ad, min_v=stats["min"], max_v=stats["max"], eps=eps)
	elif t == "quant":
     	# Prefer quantiles if available
		q01 = stats.get("q01")
		q99 = stats.get("q99")
		assert q01 is not None and q99 is not None, "Quantile-based normalization requires q01 and q99 in stats."
		return MinMaxNormalizer(action_dim=ad, min_v=stats["min"], max_v=stats["max"], q01=q01, q99=q99, eps=eps)
	elif t == "cdf":
		return CDFNormalizer(action_dim=ad, probs=stats["probs"], values=stats["values"], eps=eps)
	elif t == "identity":
		return IdentityNormalizer(action_dim=ad)
	else:
		raise ValueError(f"Unknown normalizer type: {norm_type}")
