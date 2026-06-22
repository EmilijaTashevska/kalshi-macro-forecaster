"""Property-based tests for the point-in-time query layer.

Hypothesis generates many random scenarios (including adversarial
ones) and checks that the PIT query never returns a value whose
underlying observation was unknown at the query's as_of_date.

We assert two invariants for every successful query result:

  (P1) Released-before-asof:  release_date  <= end_of_day(as_of_date)
  (P2) Vintage-before-asof:    vintage_date <= as_of_date

If P1 or P2 ever fails, we have leakage.

We also assert:

  (P3) Best-effort:  IF some observation satisfies (P1) AND (P2) for
       the queried as_of, THEN the query must return a non-None value.
       (i.e. we don't falsely refuse to return data.)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    Observation,
    SeriesDefinition,
    bulk_insert_observations,
    upsert_series_definition,
)
from kalshi_train.db.point_in_time import (
    VintagePolicy,
    _end_of_day_utc,
    pit_history,
    pit_value,
)

SERIES_ID = "TEST_SERIES"
DOMAIN_START = date(2000, 1, 1)
DOMAIN_END = date(2030, 12, 31)


@dataclass(frozen=True)
class FakeObs:
    """A simpler local triple used to drive Hypothesis examples."""

    observation_date: date
    release_date: datetime
    vintage_date: date
    value: float


# ── Hypothesis strategies ─────────────────────────────────────────────


def _date_strategy() -> st.SearchStrategy[date]:
    return st.dates(min_value=DOMAIN_START, max_value=DOMAIN_END)


def _release_datetime_for(release_day: date) -> datetime:
    # Place release at 13:30 UTC (a typical 8:30 ET economic release).
    return datetime(
        release_day.year,
        release_day.month,
        release_day.day,
        13,
        30,
        tzinfo=UTC,
    )


@st.composite
def observation_strategy(draw: st.DrawFn) -> FakeObs:
    """Generate a syntactically valid observation.

    We enforce a single sanity rule: vintage_date >= release_date (you
    cannot have a vintage of a value before that value was published).
    This is a property of valid input data, not of the leakage guard
    itself.
    """
    obs_d = draw(_date_strategy())
    rel_d = draw(_date_strategy())
    # vintage at or after release
    vint_d = draw(st.dates(min_value=rel_d, max_value=DOMAIN_END))
    value = draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    return FakeObs(
        observation_date=obs_d,
        release_date=_release_datetime_for(rel_d),
        vintage_date=vint_d,
        value=value,
    )


@st.composite
def population_strategy(draw: st.DrawFn) -> tuple[list[FakeObs], date]:
    """Generate a small population of observations + an as_of_date.

    We dedupe by (observation_date, vintage_date) since that is the
    composite primary key of series_observations.
    """
    obs = draw(st.lists(observation_strategy(), min_size=1, max_size=20))
    # Dedupe on (observation_date, vintage_date), keeping the last.
    seen: dict[tuple[date, date], FakeObs] = {}
    for o in obs:
        seen[(o.observation_date, o.vintage_date)] = o
    deduped = list(seen.values())
    as_of = draw(_date_strategy())
    return deduped, as_of


# ── Test helper: ground-truth "what should the query return?" ──────────


def _expected_observation(
    observations: list[FakeObs], as_of: date, policy: VintagePolicy
) -> FakeObs | None:
    """Reference implementation — used to verify the SQL implementation.

    A correct implementation must return the observation chosen by this
    function (or None if no observation qualifies).
    """
    as_of_eod = _end_of_day_utc(as_of.isoformat())
    qualifying = [
        o
        for o in observations
        if o.release_date.isoformat() <= as_of_eod
        and (
            policy is VintagePolicy.LATEST_REVISION
            or o.vintage_date.isoformat() <= as_of.isoformat()
        )
    ]
    if not qualifying:
        return None
    # Highest observation_date wins, then highest vintage_date.
    qualifying.sort(key=lambda o: (o.observation_date, o.vintage_date), reverse=True)
    return qualifying[0]


def _seed_db(db_path: Path, observations: list[FakeObs]) -> None:
    """Wipe + reseed the DB. Required because Hypothesis reuses the
    same `tmp_db` fixture across examples; without the wipe, data from
    earlier iterations leaks into later ones and produces spurious
    "leakage" failures.
    """
    with connect(db_path) as conn:
        conn.execute("DELETE FROM series_observations")
        conn.execute("DELETE FROM series_definitions")
        upsert_series_definition(
            conn,
            SeriesDefinition(
                series_id=SERIES_ID,
                source="TEST",
                title="Property-test series",
                frequency="monthly",
                revises=True,
            ),
        )
        bulk_insert_observations(
            conn,
            (
                Observation(
                    series_id=SERIES_ID,
                    observation_date=o.observation_date.isoformat(),
                    release_date=o.release_date.isoformat(),
                    vintage_date=o.vintage_date.isoformat(),
                    value=o.value,
                )
                for o in observations
            ),
        )
        conn.commit()


# ── Properties ────────────────────────────────────────────────────────


@settings(
    max_examples=200,
    deadline=timedelta(seconds=5),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(pop=population_strategy())
def test_no_leakage_first_known_at(pop: tuple[list[FakeObs], date], tmp_db: Path) -> None:
    """P1 + P2 + P3 under FIRST_KNOWN_AT.

    For random populations and random as_of_dates, the value returned
    must equal the value of the (release_date, vintage_date)-filtered
    most-recent observation, OR be None if no observation qualifies.
    """
    observations, as_of = pop
    _seed_db(tmp_db, observations)

    expected = _expected_observation(observations, as_of, VintagePolicy.FIRST_KNOWN_AT)
    actual = pit_value(
        SERIES_ID, as_of.isoformat(),
        policy=VintagePolicy.FIRST_KNOWN_AT,
        db_path=tmp_db,
    )

    if expected is None:
        assert actual is None, (
            f"PIT returned {actual} but no observation qualified "
            f"(as_of={as_of}, observations={observations})"
        )
    else:
        assert actual is not None, (
            f"PIT returned None but expected={expected.value} "
            f"(as_of={as_of}, observations={observations})"
        )
        # Floating compare with a small tolerance — Hypothesis uses
        # arbitrary floats.
        assert abs(actual - expected.value) < 1e-9


@settings(
    max_examples=100,
    deadline=timedelta(seconds=5),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(pop=population_strategy())
def test_returned_observation_passes_invariants(
    pop: tuple[list[FakeObs], date], tmp_db: Path
) -> None:
    """P1 + P2 directly.

    Whenever the PIT query returns a non-None value, we use
    pit_history to learn WHICH row (by its observation_date and
    vintage_date) was selected, then assert both invariants on that
    specific row. Looking up by value would be ambiguous when multiple
    observations share a value.
    """
    observations, as_of = pop
    _seed_db(tmp_db, observations)

    val = pit_value(
        SERIES_ID,
        as_of.isoformat(),
        policy=VintagePolicy.FIRST_KNOWN_AT,
        db_path=tmp_db,
    )
    if val is None:
        return

    # Identify the chosen row via pit_history at the same as_of.
    df = pit_history(
        SERIES_ID,
        as_of.isoformat(),
        as_of.isoformat(),
        policy=VintagePolicy.FIRST_KNOWN_AT,
        db_path=tmp_db,
    )
    assert not df.empty
    row = df.iloc[0]
    selected_obs_date = row["observation_date"]
    selected_vintage_date = row["vintage_date"]
    selected_release = row["release_date"]
    assert selected_obs_date is not None

    as_of_iso = as_of.isoformat()
    as_of_eod_iso = _end_of_day_utc(as_of_iso)

    # P1: the row's release_date must be on/before as_of end of day.
    assert selected_release <= as_of_eod_iso, (
        f"P1 violation: selected row release_date={selected_release} > "
        f"as_of_eod={as_of_eod_iso}"
    )
    # P2: the row's vintage_date must be on/before as_of.
    assert selected_vintage_date <= as_of_iso, (
        f"P2 violation: selected row vintage_date={selected_vintage_date} > "
        f"as_of={as_of_iso}"
    )
    # observations list itself isn't needed for invariant check; the
    # selected row's metadata already carries everything we need.
    _ = observations


@settings(
    max_examples=100,
    deadline=timedelta(seconds=5),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(pop=population_strategy())
def test_latest_revision_policy_ignores_vintage_constraint(
    pop: tuple[list[FakeObs], date], tmp_db: Path
) -> None:
    """Under LATEST_REVISION, only release_date constrains the result.

    This is by design: LATEST_REVISION is the "I know what I'm doing"
    policy that uses today's revised data. It still respects
    release_date (you can't use unreleased data even if you accept
    revision peeking).
    """
    observations, as_of = pop
    _seed_db(tmp_db, observations)

    expected = _expected_observation(observations, as_of, VintagePolicy.LATEST_REVISION)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = pit_value(
            SERIES_ID,
            as_of.isoformat(),
            policy=VintagePolicy.LATEST_REVISION,
            db_path=tmp_db,
        )

    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert abs(actual - expected.value) < 1e-9


@settings(
    max_examples=50,
    deadline=timedelta(seconds=5),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(pop=population_strategy())
def test_adversarial_unreleased_observation_never_leaks(
    pop: tuple[list[FakeObs], date], tmp_db: Path
) -> None:
    """Adversarial scenario: seed the DB with an observation released
    AFTER as_of_date and parked at a unique observation_date that no
    legitimate row in the population can occupy. The PIT query must
    never select this row.

    We use a unique observation_date (FUTURE_DOMAIN_END) as the sentinel
    identifier rather than the value field, because Hypothesis can
    randomly generate values equal to any sentinel we might pick.
    """
    observations, as_of = pop
    sentinel_obs_date = date(2099, 12, 31)
    sentinel = FakeObs(
        observation_date=sentinel_obs_date,
        release_date=_release_datetime_for(as_of + timedelta(days=30)),
        vintage_date=as_of + timedelta(days=30),
        value=999_999.0,
    )
    _seed_db(tmp_db, [*observations, sentinel])

    # We look at the chosen row's observation_date, not its value, to
    # detect leakage unambiguously.
    df = pit_history(
        SERIES_ID,
        as_of.isoformat(),
        as_of.isoformat(),
        policy=VintagePolicy.FIRST_KNOWN_AT,
        db_path=tmp_db,
    )
    selected_obs_date = df.iloc[0]["observation_date"]
    assert selected_obs_date != sentinel_obs_date.isoformat(), (
        "Leakage detected: pit selected the sentinel row whose "
        f"observation_date={sentinel_obs_date} and release_date is "
        "AFTER as_of_date."
    )


@settings(
    max_examples=50,
    deadline=timedelta(seconds=5),
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(pop=population_strategy())
def test_adversarial_future_vintage_never_leaks(
    pop: tuple[list[FakeObs], date], tmp_db: Path
) -> None:
    """Adversarial scenario: an observation whose release_date is on or
    before as_of, but whose vintage_date is AFTER as_of. The original
    vintage is fine, but the revised vintage must not be picked.

    We assert via the SELECTED row's vintage_date (a unique sentinel
    value) rather than via the float value — Hypothesis can synthesize
    any value collision otherwise.
    """
    observations, as_of = pop
    obs_date = as_of - timedelta(days=1)
    released_at = _release_datetime_for(as_of - timedelta(days=1))
    # Use a vintage_date far enough in the future that no legitimate
    # row in the population can ever have it (DOMAIN_END = 2030).
    sentinel_vintage = date(2099, 12, 31)

    original = FakeObs(
        observation_date=obs_date,
        release_date=released_at,
        vintage_date=as_of - timedelta(days=1),
        value=1.0,
    )
    future_revision = FakeObs(
        observation_date=obs_date,
        release_date=released_at,
        vintage_date=sentinel_vintage,
        value=888_888.0,
    )
    _seed_db(tmp_db, [*observations, original, future_revision])

    df = pit_history(
        SERIES_ID,
        as_of.isoformat(),
        as_of.isoformat(),
        policy=VintagePolicy.FIRST_KNOWN_AT,
        db_path=tmp_db,
    )
    if df.iloc[0]["value"] is None:
        return
    selected_vintage = df.iloc[0]["vintage_date"]
    assert selected_vintage != sentinel_vintage.isoformat(), (
        "Leakage detected: pit selected a vintage whose vintage_date "
        f"({sentinel_vintage}) is AFTER as_of_date ({as_of})."
    )
