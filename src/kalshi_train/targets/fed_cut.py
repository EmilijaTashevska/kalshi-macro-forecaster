"""Binary target: will the Fed cut the policy rate at the next FOMC meeting?

Each training row represents one FOMC meeting. We pretend to make the
prediction on the **business day before** the announcement (``as_of_date``),
using only data knowable through that date via the PIT interface.

The label is ``1`` when the upper bound of the Fed funds target range
(``DFEDTARU``) is **lower** at this meeting than at the previous meeting,
``0`` otherwise (hold or hike).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path

from kalshi_train.data.fomc_calendar import fomc_meeting_dates, previous_business_day
from kalshi_train.db.point_in_time import pit_value

DateLike = date | str

TARGET_SERIES = "DFEDTARU"
TEMPLATE_ID = "fed_decision"


@dataclass(frozen=True, slots=True)
class FedCutExample:
    """One labeled instance of the Fed-cut question."""

    meeting_date: date
    as_of_date: date
    label: int
    rate_before: float
    rate_after: float
    template_id: str = TEMPLATE_ID

    @property
    def resolution_id(self) -> str:
        return f"fed_cut_{self.meeting_date.isoformat()}"


def build_fed_cut_examples(
    *,
    start: DateLike,
    end: DateLike,
    db_path: Path | None = None,
    min_rate_history: int = 2,
) -> list[FedCutExample]:
    """Build labeled examples for every FOMC meeting in ``[start, end]``.

    Meetings where we cannot resolve ``DFEDTARU`` at the meeting date or at
    the prior meeting (via PIT) are skipped — typically the first one or two
    meetings before FRED coverage begins.
    """
    meetings = fomc_meeting_dates(start, end, db_path=db_path)
    if len(meetings) < min_rate_history:
        return []

    examples: list[FedCutExample] = []
    for prev_meeting, meeting in pairwise(meetings):
        rate_before = pit_value(TARGET_SERIES, prev_meeting, db_path=db_path)
        rate_after = pit_value(TARGET_SERIES, meeting, db_path=db_path)
        if rate_before is None or rate_after is None:
            continue

        as_of = previous_business_day(meeting)
        label = 1 if rate_after < rate_before else 0
        examples.append(
            FedCutExample(
                meeting_date=meeting,
                as_of_date=as_of,
                label=label,
                rate_before=rate_before,
                rate_after=rate_after,
            )
        )
    return examples
