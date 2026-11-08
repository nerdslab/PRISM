# PRISM-X v1 vs WDformer: where the cross-channel information enters (2026-11-08)

Code read: external/rebuttal_ctx336/WDformer/{model/WDformer.py, layers/Embed.py, layers/SelfAttention_Family.py,
layers/Transformer_EncDec.py} and variants/prism_xattn_v1_20261108/prism_xattn.py.

| | WDformer (traffic recipe) | PRISM-X v1 |
|---|---|---|
| token per channel | full Haar coefficient vector of the 336-step history (level 4: 21+21+42+84+168 coeffs), 5 Linear maps into slices of d=512; 4 calendar-mark channels are extra tokens | 6 band statistics (mean, std, amax, rough, bend, burst) averaged over the K=5 bands, computed under no_grad, LN -> Linear(6,128) -> SquaredGELU, + static channel-identity embedding |
| what crosses channels | the whole waveform content (phase, shape) of every other variate, 4 differential-attention layers with SwiGLU FFN, d=512 | 6 amplitude/roughness scalars per channel; attention output is a 128-d context vector |
| how the forecast uses it | projector Linear(512 -> H) reads the mixed token directly and is inverted by waverec: another sensor's history can be written into this sensor's future | (a) 16-d context concatenated to the router MLP input: only re-weights this channel's own bands; (b) FiLM y*(1+alpha)+beta with alpha,beta in R^H: per-step affine correction of this channel's own forecast. No path by which channel j's waveform shapes channel i's forecast |
| normalisation | instance norm per channel (mean/std over time) | SeriesNorm per channel (same idea) |
| loss / optimiser | MSE, lr 1e-3 halving every epoch (type1), batch 16, 10 epochs, patience 3 | MAE, lr 1e-4 x 0.95 per epoch, batch 64, up to 500 epochs (fast-stop batch: patience 3, delta 5e-4) |
| size at traffic | about 15 M parameters (4 layers x [QKV 3x512x1024 + out 1024x512 + SwiGLU 3x512x1024]), estimate from layer shapes | PRISM 1.3 M + 0.44 M for the context path |
| ctx-336 traffic H96 | MSE 0.353 / MAE 0.252 after 10 epochs (93 s each) | base 0.407 / 0.263 at epoch 28 (fast stop); xattn 0.409 / 0.267 at epoch 15 |

Reading of the traffic runs: xattn is ahead of base for the first 5 epochs (epoch 0 MSE 0.499 vs 0.618) and then
reaches the same plateau and stops earlier. That is what a path that can only rescale the channel's own forecast
should do: the re-weighting it can express is learned fast, but it brings no information the channel did not
already have. WDformer's advantage on traffic MSE comes from (1) mixing waveform content across the 862 sensors
and reading the forecast out of the mixed representation, (2) the MSE objective (it loses MAE on every horizon),
and (3) explicit calendar tokens. v1 reproduces none of these three.

Possible v2 (not run): make the context carry waveform content (per-channel band coefficients or PRISM's own band
representations projected to d) and add a zero-initialised residual read-out Linear(d -> H) to the forecast, so
other channels can contribute shape, not only scale; keep MAE for the MAE comparison, MSE for the MSE one.
