# Glossary

Short definitions of every term used in these documents, in the sense we use
them here. Alphabetical.

**Action counterfactual.** Dreaming from the same starting state under two
different action sequences and comparing the outcomes. If the imagined futures
differ in the right way, the model has learned that actions have effects.

**Active unit.** A latent dimension of the VAE whose KL to the prior is well
above zero, i.e. one that actually carries information. Inactive units sit at
exactly `μ=0, σ=1` and could be deleted without changing any reconstruction.

**Autocorrelation (of frames).** Consecutive frames in an episode are nearly
identical, so 200 frames from one episode are far fewer than 200 independent
samples. This is why probe sets use many short episodes and why train/test
splits are made at the episode level.

**CMA-ES.** Covariance Matrix Adaptation Evolution Strategy. A black-box
optimiser: sample a population of parameter vectors from a Gaussian, evaluate
each, move the Gaussian toward the better ones, repeat. Needs no gradients,
which is why it can train a policy through a non-differentiable simulator (or a
sampled dream). Only practical for small parameter counts, hence the tiny C.

**Controller (C).** The policy. Here a single linear layer from `[z, h]` to
three action scores. Deliberately tiny so CMA-ES can train it.

**Dream / rollout / imagination.** Running M forward on its *own* predictions:
feed the sampled `z_{t+1}` back in as the next input, never looking at a real
frame. Decoding each dreamed `z` with V's decoder produces an imagined video.

**Free bits.** A modification of the VAE objective: each latent dimension's KL is
replaced by `max(KL_i, λ)`, so the first λ nats of information per dimension
are "free". Prevents posterior collapse by removing the gradient that would
crush a dimension to the prior before the decoder has learned to use it.

**Hidden state (h).** The recurrent network's memory vector, carried from step
to step. Anything about the world that is not visible in the current frame
(velocity, what happened recently) has to be stored here.

**KL divergence.** `KL(q || p)`: how much information is lost when `p` is used
to approximate `q`, in nats. In a VAE it measures how far the encoder's
posterior for one frame is from the standard-normal prior, and acts as the
"information budget" the encoder pays to transmit `z`.

**kNN probe.** A k-nearest-neighbours regressor from latents to a true state
variable. Nonparametric, so it answers "is the information present under *any*
smooth map?" — the upper bound among probes.

**Latent / code (z).** The compact vector V produces for a frame. 16 numbers
here.

**Linear probe.** Ordinary least squares from latents to a true variable. High
R² means the variable is encoded as a linear direction in latent space; low R²
with high kNN R² means it is present but encoded nonlinearly.

**MCC (Mean Correlation Coefficient).** Match each true factor to one latent
dimension (Hungarian algorithm) to maximise total |correlation|, then average.
The identifiability literature's summary metric. Blind to non-monotone codes.

**MDN (Mixture Density Network).** An output layer that predicts the parameters
of a *mixture of Gaussians* — mixing weights, means, standard deviations —
instead of a single value. Lets the model say "the ball will be either here or
there" at a bounce, rather than averaging to a wrong middle.

**MDN-RNN (M).** An LSTM whose output head is an MDN over the next latent. The
"memory" component of the world model.

**Nat.** Unit of information using natural log; 1 nat ≈ 1.44 bits.

**NLL (negative log-likelihood).** `−log p(target)` under the model's predicted
distribution. Lower is better. The training loss for the MDN.

**Open-loop vs teacher-forced.** *Teacher-forced*: at every step the model is
given the true `z_t` and only has to predict one step ahead. *Open-loop*: the
model is given its own previous prediction. Teacher-forced error is what you
train on; open-loop error is what you actually care about, and it compounds.

**Place-field code.** A latent representation where individual dimensions act
like detectors for "object is in region R" rather than as coordinates. Named
after hippocampal place cells. Positions are still fully recoverable from the
whole vector, but not from any single dimension or any linear readout.

**Posterior collapse.** The failure mode where the VAE encoder outputs the prior
for every input (`μ→0`), the decoder learns to ignore `z`, and the model
reconstructs the mean image. Self-reinforcing once it happens.

**Probe.** Any regressor fit from learned representations to ground-truth
variables, used to ask what information a representation contains and in what
form. Never used as a training signal.

**R².** Coefficient of determination: fraction of a variable's variance a
predictor explains. 1 = perfect, 0 = no better than the mean, negative = worse
than the mean (possible on held-out data).

**Reparameterisation trick.** Writing a sample as `z = μ + σ·ε` with `ε ~ N(0,1)`
so gradients flow into `μ` and `σ` even though `z` is random.

**Sticky actions.** Holding a randomly chosen action for a geometrically
distributed number of steps instead of re-sampling every step. Produces long
paddle sweeps and a much wider distribution over paddle position/velocity.

**Temperature (τ).** A knob on the MDN's sampling: multiply the predicted
standard deviations by τ (and sharpen the mixture weights by 1/τ). τ→0 gives
the single most likely prediction; τ=1 samples from the model's true belief;
τ>1 makes the dream noisier than the model believes. Ha & Schmidhuber train C
at τ>1 to stop it exploiting an over-confident dream.

**Tuning map.** For one latent dimension, its mean value as a function of the
true ball position, drawn as a heat-map over the box. Shows *what* the dimension
computes: a ramp (coordinate), a blob (place field), stripes (periodic).

**Useful dream horizon.** Our name for the number of dreamed steps before the
decoded ball position drifts by more than one ball radius from the truth. A
single number for "how far ahead can the model imagine."

**VAE (Variational Autoencoder).** An autoencoder whose encoder outputs a
distribution and whose loss includes a KL term to a prior. The "vision"
component V.

**World model.** A learned model of an environment's dynamics that can be used
to predict, imagine, plan, or train a policy without (or with less) real
interaction.
