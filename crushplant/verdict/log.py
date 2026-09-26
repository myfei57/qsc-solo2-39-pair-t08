"""The verdict trail: what the current state says and what history recorded."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from ..clock import parse_stamp
from ..errors import InvalidRequest
from ..store.records import RecordStream

VERDICT_KIND = "verdict"


@dataclass(frozen=True)
class VerdictEntry:
    """One judgement as it was written to the record stream.

    ``generation`` is the parameter generation the judgement was made under;
    a zero means the record predates generation tracking and cannot be placed.
    """

    sequence: int
    at: str
    unit: str
    subject: str
    name: str
    state: str
    value: float
    generation: int
    actor: str
    detail: dict[str, Any]

    def moment(self) -> datetime:
        return parse_stamp(self.at)

    def ok(self) -> bool:
        return self.state == "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "at": self.at,
            "unit": self.unit,
            "subject": self.subject,
            "name": self.name,
            "state": self.state,
            "value": self.value,
            "generation": self.generation,
            "actor": self.actor,
            "detail": self.detail,
        }


class VerdictLog:
    """Writes every judgement to the stream and reads the newest one back.

    ``current`` answers "what does the line say now", while ``history`` answers
    "what did it say along the way".  The two are deliberately different calls
    so a report cannot present an archived judgement as a live one.  Every
    record is stamped with the parameter generation in force when it was
    written, so a later judgement never has to overwrite an earlier one for
    the pair to be told apart.
    """

    def __init__(self, stream: RecordStream, generation_of: Callable[[], int] | None = None) -> None:
        self._stream = stream
        self._generation_of = generation_of or (lambda: 0)

    @property
    def stream(self) -> RecordStream:
        return self._stream

    def record(
        self,
        unit: str,
        subject: str,
        name: str,
        state: str,
        value: float,
        moment: datetime,
        actor: str,
        **detail: Any,
    ) -> VerdictEntry:
        if not name.strip():
            raise InvalidRequest("a verdict needs a name")
        payload = {
            "name": name.strip(),
            "state": state,
            "value": float(value),
            "generation": self._generation_of(),
            "detail": dict(detail),
        }
        record = self._stream.stage(
            VERDICT_KIND,
            moment,
            unit=unit,
            actor=actor,
            subject=subject.strip(),
            payload=payload,
        )
        self._stream.commit_through(record.sequence, moment, actor)
        return self._entry(record)

    def record_threshold(
        self,
        unit: str,
        verdict: Any,
        moment: datetime,
        actor: str,
        **detail: Any,
    ) -> VerdictEntry:
        """Write a :class:`~crushplant.verdict.threshold.Verdict` down as it stands."""

        return self.record(
            unit,
            verdict.subject,
            verdict.name,
            verdict.state,
            verdict.value,
            moment,
            actor,
            basis=verdict.name,
            low=verdict.low,
            high=verdict.high,
            margin=verdict.margin,
            **detail,
        )

    def record_window(
        self,
        unit: str,
        verdict: Any,
        moment: datetime,
        actor: str,
        **detail: Any,
    ) -> VerdictEntry:
        """Write a :class:`~crushplant.verdict.window.WindowVerdict` down."""

        return self.record(
            unit,
            verdict.subject,
            f"{verdict.subject}.window",
            verdict.state,
            verdict.latest,
            moment,
            actor,
            basis=f"{verdict.subject}.window",
            threshold=verdict.threshold,
            samples=verdict.samples,
            span_seconds=verdict.span_seconds,
            **detail,
        )

    def entries(
        self,
        subject: str = "",
        unit: str = "",
        generation: int | None = None,
        limit: int | None = None,
    ) -> list[VerdictEntry]:
        selected = [
            self._entry(record)
            for record in self._stream.visible()
            if record.kind == VERDICT_KIND
            and (not subject or record.subject == subject)
            and (not unit or record.unit == unit)
        ]
        if generation is not None:
            selected = [entry for entry in selected if entry.generation == int(generation)]
        if limit is None:
            return selected
        if limit < 0:
            raise InvalidRequest("a verdict limit must not be negative")
        return selected[-limit:] if limit else []

    def history(self, subject: str = "") -> list[VerdictEntry]:
        return self.entries(subject=subject)

    def current(self, subject: str = "") -> dict[str, VerdictEntry]:
        """The newest judgement per subject, which is what the line says now."""

        newest: dict[str, VerdictEntry] = {}
        for entry in self.entries(subject=subject):
            newest[entry.subject] = entry
        return newest

    def as_of(self, moment: datetime, subject: str = "") -> dict[str, VerdictEntry]:
        """The newest judgement per subject as the trail stood at ``moment``.

        This is the shift-handover view: what the line had last said about
        each subject at an earlier instant, rebuilt from the records that were
        already written by then.
        """

        newest: dict[str, VerdictEntry] = {}
        for entry in self.entries(subject=subject):
            if entry.moment() <= moment:
                newest[entry.subject] = entry
        return newest

    def by_basis(self, basis: str, unit: str = "") -> list[VerdictEntry]:
        """Every judgement taken on one basis, newest last."""

        label = basis.strip()
        if not label:
            raise InvalidRequest("a basis query needs a name")
        return [entry for entry in self.entries(unit=unit) if entry.detail.get("basis") == label]

    def bases(self) -> list[str]:
        """The distinct bases the recorded judgements were taken on."""

        return sorted({str(entry.detail["basis"]) for entry in self.entries() if "basis" in entry.detail})

    def generations(self) -> list[int]:
        """The parameter generations the recorded judgements were made under."""

        return sorted({entry.generation for entry in self.entries()})

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries():
            counts[entry.state] = counts.get(entry.state, 0) + 1
        return counts

    def summary(self) -> dict[str, Any]:
        return {
            "recorded": len(self.entries()),
            "subjects": sorted(self.current()),
            "states": self.counts(),
            "bases": self.bases(),
        }

    @staticmethod
    def _entry(record: Any) -> VerdictEntry:
        payload = record.payload
        detail = payload.get("detail")
        return VerdictEntry(
            sequence=record.sequence,
            at=record.at,
            unit=record.unit,
            subject=record.subject,
            name=str(payload.get("name", "")),
            state=str(payload.get("state", "")),
            value=float(payload.get("value", 0.0)),
            generation=int(payload.get("generation", 0)),
            actor=record.actor,
            detail=dict(detail) if isinstance(detail, dict) else {},
        )
