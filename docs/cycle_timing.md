# Cycle timing

Scope v2.1 §6.2. What changed from v2, and the arithmetic behind the numbers.

## The v2 problem

v2's schedule had AEOLUS products ready at t+0:45, consuming the GFS cycle named
t. GFS 0.25 degree publishes roughly 3.5-4 hours after its cycle time. At
t+0:45 that data does not exist. The schedule was not tight — it was impossible.

## The v2.1 cycle

```
t+0:00   prefetch begins against already-staged t-6 NWP fields
t+0:45   TC-Vitals nominally arrives; cycle formally starts
t+1:20   vitals timeout: run on an extrapolated fix, flag vitals=estimated
t+1:40   products ready (nominal path)
t+2:27   products ready (worst case, standard profile)
t+3:00   NHC public advisory
```

Two changes carry it:

**Cycle t consumes the t-6 NWP cycle.** The t-6 cycle's 6-hour forecast is valid
at t and published well before t. `select_nwp_cycle` will not select the cycle
named t even if it is offered, because at t it does not exist and selecting it
silently is precisely the v2 error.

**The cycle is gated on the working fix, not the clock.** The storm's current
position is what the whole forecast is anchored to. Starting without it produces
a forecast of a storm that is not where you think it is.

## Deriving the timeout

The scope wrote the vitals timeout as a flat t+1:30. That does not survive the
arithmetic. With the reduced profile's 70-minute worst case and the 30-minute
advisory margin:

```
1:30 start + 1:10 worst case = 2:40 finish, leaving 20 minutes. Short by 10.
```

`derive_vitals_timeout` computes it instead:

```
timeout = advisory_offset - reduced_worst_case - margin
        = 3:00 - 1:10 - 0:30
        = 1:20
```

Deriving rather than asserting keeps the three numbers consistent when any of
them is retuned. Change a stage budget and the timeout follows.

## Load shedding

When the cycle starts late enough that the standard profile would breach the
margin, `plan_cycle` switches to `REDUCED_BUDGETS` and sets `load_shed`. The
saving is almost entirely MERIDIAN: roughly 10 ensemble members instead of
20-50.

That is a real loss of tail resolution. It is a smaller loss than delivering
nothing before the advisory goes out. `allow_load_shedding=False` disables it,
and `test_load_shedding_can_be_disabled_and_then_the_deadline_is_missed`
documents the trade explicitly rather than leaving it implied.

## Degraded modes

| Condition | Behaviour | Flag |
|---|---|---|
| Working fix late past timeout | Extrapolate from the last two fixes | `vitals=estimated` |
| t-6 NWP missing | Fall back to t-12 | `nwp_stale=12h` |
| All NWP missing | Abandon; LSTM + climatology mode | `CycleAbandoned` |
| Diffusion crash | Climatological-spread ensemble | `meridian_fallback:*` |
| Optional feed missing | Continue | `missing:<source>` |
| Opportunistic feed missing | Continue, no flag | — |
| Late start | Reduced ensemble | `load_shed` |

Extrapolation is linear on the last motion vector with intensity held constant.
Over the ~45 minutes being bridged, a cleverer intensity model adds variance
without adding skill.

## What is not yet wired

`CyclePlan.load_shed` is set and flagged, but `run_cycle` does not yet reduce
the requested member count. That connection should be made before any live use.
