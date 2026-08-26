# Model-set pinning

Scope v2.1 §5.7. Why promoting a single model is not a safe operation.

## The dependency

MERIDIAN's diffusion model and the fusion consensus layer do not train on raw
data. They train on **latents extracted from specific Group 1 checkpoints**
(LSTM, CNN, Transformer, GNN, PINN).

v2's drift trigger said: retrain the affected model only. Apply that to an LSTM
and you get a new LSTM producing latents in a different representation space,
feeding a diffusion model trained to interpret the old one. Nothing errors. The
ensemble simply becomes worse, in a way that is very hard to attribute.

## The mechanism

`latent_signature()` builds a deterministic signature from the Group 1 version
set:

```
lstmv3-cnnv2-transformerv5-gnnv2-pinnv1
```

Derived models record the signature they were trained against — `register`
refuses a derived model without one. `pin_set` then requires:

- every model in the system is present
- all members are operational-flavor
- every derived model's signature matches the pinned Group 1 set

Bump one Group 1 version and the signature changes, so the pin is refused until
latents are regenerated and the derived models retrained. The invariant cannot
be violated by forgetting.

## The trigger side

`triggers.expand_jobs` applies the same rule going the other way: any Group 1
retrain job automatically expands to include diffusion and fusion, marked
`cascaded=True`, ordered after their dependencies. `on_drift("lstm")` returns
three jobs, not one.

## Nightly refresh suppression

v2 retrained the diffusion model nightly at 02:00, unconditionally. During an
active storm that means consecutive advisories carry forecasts from different
models, and any change in ensemble spread between them is uninterpretable — you
cannot tell whether uncertainty grew or the generator changed.

`nightly_latent` returns no jobs while `SeasonState.storms_active` is true. The
pinned set holds for the duration of the storm.

## Rollback

`ModelRegistry.rollback` returns production to the most recent archived version.
Registry state persists to JSON, so rollback survives a process restart — which
matters, because the moment you need it is unlikely to be a calm one.

MLflow, when configured, is mirrored best-effort: any exception from the client
disables the mirror and the local store continues. Tracking must never fail a
training run.
