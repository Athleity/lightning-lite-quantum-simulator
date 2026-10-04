"""Neural decoder for the surface code memory experiment (hybrid MLP + MWPM).

Physics
-------
A decoder maps the detector syndrome s in {0,1}^D of a memory circuit to a logical
correction c in {0,1}^K. MWPM decodes an independent-error graph model of the circuit.
An exact (maximum-likelihood) decoder would pick argmax_c P(c | s) under the true circuit
noise, including correlated faults (Y errors, hook errors, correlated two-qubit faults)
that the matching graph only approximates. A neural network trained on sampled
(syndrome, observable flip) pairs estimates P(c | s) directly:

    logit_k(s) = MLP(s, m(s))_k + alpha_k (2 m_k(s) - 1),      P(c_k = 1 | s) = sigmoid(logit_k)

where m(s) is the MWPM prediction. The input is the syndrome plus m(s), and the second
term is a residual path with learnable gain alpha_k, initialised to 4 and with the MLP's
last layer initialised near zero, so training starts at (a recalibrated) MWPM and can only
move away from it where the data supports it. Loss is binary cross-entropy per observable.
Decoding picks c_k = 1 when logit_k > 0.

Data scale. The syndrome space has 2^D states (D ~ 24 at d = 3 with 3 rounds, over 100
at d = 5), so the network sees a vanishing fraction of it at larger d. It generalises
through local structure, not by memorising. Expect d = 3 to work, d = 5 to be competitive
at best, d = 7 to be at or below MWPM unless the training set is far larger than 10^7.
Google's Willow decoder is a recurrent transformer trained on far more data. This MLP is a
small stand-in, and the benchmark should report it as such.

The training circuit is built through SurfaceCodeMemory(distance, p, rounds), so detector
order and noise are identical to the circuit the decoder is tested on. A decoder is valid
only for the (distance, rounds, p, noise) it was trained on.

Validate against (self-test): train and validation loss fall on 10k shots; on d = 3,
p = 0.003 the decoder is no worse than MWPM on a shared test set (paired z <= 2), with
a separate printout of whether it is significantly better; two decoders with the same seed
give identical weights and predictions.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

try:
    import stim
except ImportError as exc:
    raise ImportError("NeuralDecoder needs stim: pip install stim pymatching") from exc

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("NeuralDecoder needs PyTorch: pip install torch") from exc

try:
    from SurfaceCode import DEFAULT_P, SurfaceCodeMemory, make_decoder
except ImportError:  # imported from another working directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from SurfaceCode import DEFAULT_P, SurfaceCodeMemory, make_decoder

__all__ = ["NeuralDecoder"]

DEFAULT_HIDDEN = 128
MIN_HIDDEN = 16
SAMPLE_CHUNK = 100_000     # shots sampled and MWPM-decoded per chunk (bounds memory)
INFER_CHUNK = 65_536
ALPHA_INIT = 4.0           # initial gain on the MWPM residual path
LAST_LAYER_STD = 1e-2


def _circuit_of(memory) -> stim.Circuit:
    """The stim circuit a SurfaceCodeMemory samples from."""
    for name in ("circuit", "_circuit", "build_circuit", "make_circuit", "memory_circuit"):
        v = getattr(memory, name, None)
        if callable(v):
            try:
                v = v()
            except TypeError:
                continue
        if isinstance(v, stim.Circuit):
            return v
    raise AttributeError("could not find the stim circuit on SurfaceCodeMemory; "
                         "edit _circuit_of in NeuralDecoder.py to return it")


class _Net(nn.Module):
    def __init__(self, n_in: int, hidden: int, n_obs: int, dropout: float, hybrid: bool):
        super().__init__()
        self.n_obs, self.hybrid = n_obs, hybrid
        self.mlp = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_obs))
        nn.init.normal_(self.mlp[-1].weight, std=LAST_LAYER_STD)
        nn.init.zeros_(self.mlp[-1].bias)
        if hybrid:
            self.alpha = nn.Parameter(torch.full((n_obs,), ALPHA_INIT))

    def forward(self, x):
        z = self.mlp(x)
        if self.hybrid:
            z = z + self.alpha * (2.0 * x[:, -self.n_obs:] - 1.0)
        return z


class NeuralDecoder:
    """MLP decoder trained on stim syndromes of one surface code memory experiment.

    Implements the Decoder protocol of SurfaceCode.py: decode_batch(detections) returns
    the predicted observable flips, shape (shots, num_observables), dtype uint8.
    hybrid=True (default) adds the MWPM prediction as an input and a residual path.
    `hidden` sets both hidden layers. Raises ValueError for invalid arguments.
    """

    def __init__(self, distance: int, rounds: int | None = None, p: float = DEFAULT_P,
                 hidden: int = DEFAULT_HIDDEN, seed: int = 0, hybrid: bool = True,
                 dropout: float = 0.0):
        if hidden < MIN_HIDDEN:
            raise ValueError(f"hidden must be >= {MIN_HIDDEN}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not 0.0 < p <= 1.0:
            raise ValueError("p must lie in (0, 1]")
        self.memory = SurfaceCodeMemory(distance, p, rounds)
        self.distance, self.p = distance, p
        self.rounds = getattr(self.memory, "rounds", rounds)
        self.hidden, self.seed, self.hybrid, self.dropout = hidden, seed, hybrid, dropout

        self.circuit = _circuit_of(self.memory)
        self.num_detectors = self.circuit.num_detectors
        self.num_observables = self.circuit.num_observables
        if self.num_detectors == 0 or self.num_observables == 0:
            raise ValueError("circuit has no detectors or observables (p = 0?)")
        dem = self.circuit.detector_error_model(decompose_errors=True)
        self._mwpm = make_decoder("mwpm", dem)

        torch.manual_seed(seed)
        n_in = self.num_detectors + (self.num_observables if hybrid else 0)
        self.net = _Net(n_in, hidden, self.num_observables, dropout, hybrid)
        self.trained = False
        self.history: dict = {}
        self.train_seconds = 0.0
        self.last_inference_seconds = 0.0

    @property
    def name(self) -> str:
        return "neural"

    # -- data ------------------------------------------------------------
    def _mwpm_bits(self, det: np.ndarray) -> np.ndarray:
        return np.asarray(self._mwpm.decode_batch(det), dtype=np.uint8).reshape(
            det.shape[0], self.num_observables)

    def _features(self, det: np.ndarray) -> np.ndarray:
        d = det.astype(np.uint8)
        return np.concatenate([d, self._mwpm_bits(det)], axis=1) if self.hybrid else d

    def sample(self, n: int, seed: int):
        """(features uint8, observable flips uint8) for n shots of the training circuit."""
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        feats, obs = [], []
        done = 0
        while done < n:
            m = min(SAMPLE_CHUNK, n - done)
            det, ob = sampler.sample(m, separate_observables=True)
            feats.append(self._features(det))
            obs.append(ob.astype(np.uint8))
            done += m
        return np.concatenate(feats), np.concatenate(obs)

    # -- training --------------------------------------------------------
    def _eval(self, F, Y, loss_fn):
        self.net.eval()
        tot, err = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(F), INFER_CHUNK):
                x = torch.from_numpy(F[i:i + INFER_CHUNK].astype(np.float32))
                y = torch.from_numpy(Y[i:i + INFER_CHUNK].astype(np.float32))
                z = self.net(x)
                tot += float(loss_fn(z, y)) * len(x)
                err += int(((z > 0) != (y > 0.5)).any(dim=1).sum())
        return tot / len(F), err / len(F)

    def train(self, n_samples: int = 1_000_000, epochs: int = 20, batch: int = 256,
              lr: float = 1e-3, val_fraction: float = 0.1, verbose: bool = True) -> dict:
        """Train on n_samples fresh shots. Returns history with train_loss, val_loss and
        val_error per epoch (index 0 is the untrained network). The last val_fraction of
        the shots is held out and never trained on."""
        if n_samples < 100 or epochs < 1 or batch < 1 or lr <= 0.0:
            raise ValueError("need n_samples >= 100, epochs >= 1, batch >= 1, lr > 0")
        if not 0.0 < val_fraction < 0.5:
            raise ValueError("val_fraction must lie in (0, 0.5)")
        t0 = time.perf_counter()
        F, Y = self.sample(n_samples, seed=self.seed + 1)
        n_val = max(1, int(n_samples * val_fraction))
        Ftr, Ytr, Fva, Yva = F[:-n_val], Y[:-n_val], F[-n_val:], Y[-n_val:]
        loss_fn = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        gen = torch.Generator().manual_seed(self.seed)

        hist = {"train_loss": [], "val_loss": [], "val_error": []}
        tl, _ = self._eval(Ftr[:n_val], Ytr[:n_val], loss_fn)
        vl, ve = self._eval(Fva, Yva, loss_fn)
        hist["train_loss"].append(tl), hist["val_loss"].append(vl), hist["val_error"].append(ve)
        if verbose:
            print(f"  epoch  0  train {tl:.5f}  val {vl:.5f}  val err {ve:.4e}  (untrained)")
        for ep in range(1, epochs + 1):
            self.net.train()
            perm = torch.randperm(len(Ftr), generator=gen).numpy()
            run, seen = 0.0, 0
            for i in range(0, len(perm), batch):
                idx = np.sort(perm[i:i + batch])
                x = torch.from_numpy(Ftr[idx].astype(np.float32))
                y = torch.from_numpy(Ytr[idx].astype(np.float32))
                opt.zero_grad()
                loss = loss_fn(self.net(x), y)
                loss.backward()
                opt.step()
                run += float(loss.detach()) * len(idx)
                seen += len(idx)
            sched.step()
            vl, ve = self._eval(Fva, Yva, loss_fn)
            hist["train_loss"].append(run / seen), hist["val_loss"].append(vl)
            hist["val_error"].append(ve)
            if verbose:
                print(f"  epoch {ep:2d}  train {run / seen:.5f}  val {vl:.5f}  val err {ve:.4e}")
        self.trained = True
        self.history = hist
        self.train_seconds = time.perf_counter() - t0
        if verbose:
            print(f"  trained on {len(Ftr)} shots in {self.train_seconds:.1f} s")
        return hist

    # -- Decoder protocol --------------------------------------------------
    def decode_batch(self, detections: np.ndarray) -> np.ndarray:
        """Predicted observable flips, shape (shots, num_observables), uint8."""
        if not self.trained:
            raise RuntimeError("NeuralDecoder.decode_batch: call train() first")
        det = np.asarray(detections)
        if det.ndim != 2 or det.shape[1] != self.num_detectors:
            raise ValueError(f"detections must have shape (shots, {self.num_detectors}); "
                             f"got {det.shape}")
        t0 = time.perf_counter()
        self.net.eval()
        out = np.empty((det.shape[0], self.num_observables), dtype=np.uint8)
        with torch.no_grad():
            for i in range(0, det.shape[0], INFER_CHUNK):
                f = self._features(det[i:i + INFER_CHUNK])
                z = self.net(torch.from_numpy(f.astype(np.float32)))
                out[i:i + INFER_CHUNK] = (z > 0).numpy().astype(np.uint8)
        self.last_inference_seconds = time.perf_counter() - t0
        return out

    def weights(self) -> np.ndarray:
        return np.concatenate([p.detach().numpy().ravel() for p in self.net.parameters()])


def _self_test() -> None:
    # 1. Loss falls on 10k shots (index 0 is the untrained network).
    nd = NeuralDecoder(3, 3, 0.003, hidden=64, seed=0)
    h = nd.train(n_samples=10_000, epochs=8, batch=128, verbose=False)
    assert h["train_loss"][-1] < h["train_loss"][0], h["train_loss"]
    assert h["val_loss"][-1] < h["val_loss"][0], h["val_loss"]
    print(f"[1] 10k shots: train loss {h['train_loss'][0]:.4f} -> {h['train_loss'][-1]:.4f}, "
          f"val loss {h['val_loss'][0]:.4f} -> {h['val_loss'][-1]:.4f}, "
          f"val error {h['val_error'][0]:.4e} -> {h['val_error'][-1]:.4e}")

    # 2. d = 3, p = 0.003: no worse than MWPM on a shared test set (paired comparison).
    nd2 = NeuralDecoder(3, 3, 0.003, hidden=128, seed=0)
    nd2.train(n_samples=400_000, epochs=6, batch=512, verbose=False)
    sampler = nd2.circuit.compile_detector_sampler(seed=987654)
    det, obs = sampler.sample(200_000, separate_observables=True)
    mw_wrong = (nd2._mwpm_bits(det) != obs).any(axis=1)
    nn_wrong = (nd2.decode_batch(det) != obs).any(axis=1)
    b = int(np.count_nonzero(nn_wrong & ~mw_wrong))   # neural wrong, MWPM right
    c = int(np.count_nonzero(~nn_wrong & mw_wrong))   # MWPM wrong, neural right
    z = (b - c) / max(np.sqrt(b + c), 1.0)
    n = len(obs)
    print(f"[2] d=3 p=0.003: MWPM {mw_wrong.mean():.4e}, neural {nn_wrong.mean():.4e} "
          f"({n} shots); discordant: neural-only wrong {b}, MWPM-only wrong {c}, z = {z:+.2f}")
    print("    neural significantly BETTER than MWPM" if z < -2.0 else
          "    neural not significantly better than MWPM")
    assert z <= 2.0, f"neural decoder significantly worse than MWPM (z = {z:.2f})"

    # 3. Reproducible with the same seed.
    a = NeuralDecoder(3, 3, 0.003, hidden=32, seed=7)
    bdec = NeuralDecoder(3, 3, 0.003, hidden=32, seed=7)
    a.train(n_samples=5_000, epochs=3, batch=128, verbose=False)
    bdec.train(n_samples=5_000, epochs=3, batch=128, verbose=False)
    assert np.array_equal(a.weights(), bdec.weights()), "weights differ for equal seeds"
    probe = nd2.circuit.compile_detector_sampler(seed=5).sample(2000, separate_observables=True)[0]
    assert np.array_equal(a.decode_batch(probe), bdec.decode_batch(probe))
    print("[3] same seed: identical weights and predictions")

    # Input checks.
    for bad in (lambda: NeuralDecoder(3, 3, 0.003, hidden=4),
                lambda: NeuralDecoder(3, 3, 0.0),
                lambda: NeuralDecoder(3, 3, 0.003, dropout=1.0),
                lambda: NeuralDecoder(3, 3, 0.003).decode_batch(probe),
                lambda: nd2.decode_batch(probe[:, :-1]),
                lambda: nd2.train(n_samples=10)):
        try:
            bad()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("expected an exception")
    print("NeuralDecoder: self-test passed")


if __name__ == "__main__":
    _self_test()