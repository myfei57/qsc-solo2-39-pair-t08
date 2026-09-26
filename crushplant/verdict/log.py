"""The verdict trail: what the current state says and what history recorded."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from ..clock import parse_stamp
from ..errors import InvalidRequest, RecordNotFound
from ..store.records import RecordStream

VERDICT_KIND = "verdict"


@dataclass(frozen=True)
class VerdictEntry:
    """One judgement as it was written to the record stream.

    ``generation`` is the parameter generation that was live when the judgement
    was taken and ``basis`` is the limit name (or rule name) it was taken
    against.  Both are snapshots fixed on the record itself, so a later
    generation bump can never change what an older judgement was decided on.
    """

    sequence: int
    at: str
    unit: str
    subject: str
    name: str
    state: str
    value: float
    actor: str
    detail: dict[str, Any]
    generation: int = 0
    basis: str = ""
    low: float | None = None
    high: float | None = None
    margin: float | None = None
    voided: bool = False
    void_reason: str = ""

    def moment(self) -> datetime:
        return parse_stamp(self.at)

    def ok(self) -> bool:
        return self.state == "ok"

    def breached(self) -> bool:
        """Whether the reading sat outside its window when this was judged.

        A verdict is a breach when its state is anything other than ``ok`` or
        when, for a verdict that carried bounds, the margin is negative.  The
        answer is computed from this record alone, never from the current
        limits, so an archived verdict keeps the answer it had at the time.
        """

        if self.state != "ok":
            return True
        return self.margin is not None and self.margin < 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "at": self.at,
            "unit": self.unit,
            "subject": self.subject,
            "name": self.name,
            "state": self.state,
            "value": self.value,
            "actor": self.actor,
            "generation": self.generation,
            "basis": self.basis,
            "low": self.low,
            "high": self.high,
            "margin": self.margin,
            "breached": self.breached(),
            "voided": self.voided,
            "void_reason": self.void_reason,
            "detail": self.detail,
        }


class VerdictLog:
    """Writes every judgement to the stream and reads them back.

    ``current`` answers "what does the line say now", while ``history`` answers
    "what did it say along the way".  The two are deliberately different calls
    so a report cannot present an archived judgement as a live one.  Nothing
    here ever rewrites a past entry: a newer judgement for the same subject is
    appended alongside the older ones, and the parameter generation, basis and
    bounds that were in force are copied onto the record before it is
    committed, which is what a shift hand-over reads the trail against.
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
        # Snapshot the parameter generation on the way in.  A caller may pass
        # one explicitly (a reading judged against a captured baseline carries
        # the generation that baseline belongs to); otherwise the live
        # generation at the instant of judgement is frozen onto the record.
        generation = int(detail.pop("generation", self._generation_of()) or 0)
        basis = str(detail.pop("basis", "") or name.strip())
        bounds: dict[str, Any] = {}
        for key in ("low", "high", "margin"):
            if key in detail:
                bounds[key] = float(detail.pop(key))
        payload = {
            "name": name.strip(),
            "state": state,
            "value": float(value),
            "generation": generation,
            "basis": basis,
            "detail": dict(detail),
            **bounds,
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
        *,
        basis: str = "",
        generation: int | None = None,
        breached: bool | None = None,
        limit: int | None = None,
        include_voided: bool = False,
    ) -> list[VerdictEntry]:
        """Return recorded judgements, oldest first.

        Every filter narrows the result and an unset one places no condition.
        Voided records are excluded by default so the live trail stays clean;
        pass ``include_voided`` to read them with their tombstone annotation
        for a hand-over review.
        """

        selected: list[VerdictEntry] = []
        voided = self._stream.voided_sequences()
        for record in self._stream.committed():
            if record.kind != VERDICT_KIND or record.is_void:
                continue
            is_voided = record.sequence in voided
            if is_voided and not include_voided:
                continue
            entry = self._entry(record, voided=voided)
            if subject and entry.subject != subject:
                continue
            if unit and entry.unit != unit:
                continue
            if basis and entry.basis != basis:
                continue
            if generation is not None and entry.generation != int(generation):
                continue
            if breached is not None and entry.breached() != breached:
                continue
            selected.append(entry)
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
        """The newest judgement each subject held at or before an instant.

        Records stamped after ``moment`` are ignored, so this reconstructs
        what an operator on the earlier shift would have seen as current
        without touching or replacing any later judgement.
        """

        point = moment.timestamp()
        newest: dict[str, VerdictEntry] = {}
        for entry in self.entries(subject=subject):
            if entry.moment().timestamp() <= point:
                newest[entry.subject] = entry
        return newest

    def by_basis(self, basis: str, subject: str = "", unit: str = "") -> list[VerdictEntry]:
        """Every judgement taken against one named limit or rule."""

        return self.entries(subject=subject, unit=unit, basis=basis)

    def get(self, sequence: int) -> VerdictEntry:
        """Fetch one live judgement by its immutable stream sequence.

        A tombstoned record is not handed back as a live judgement; read the
        full trail with ``include_voided`` to review it and its reason.
        """

        record = self._stream.journal.require(sequence)
        if record.kind != VERDICT_KIND or record.is_void:
            raise RecordNotFound("that sequence is not a verdict", sequence=sequence)
        if sequence in self._stream.voided_sequences():
            raise RecordNotFound("that verdict was tombstoned", sequence=sequence)
        return self._entry(record)

    def bases(self) -> list[str]:
        """The distinct judgement rules the trail was taken against, sorted."""

        return sorted({entry.basis for entry in self.entries() if entry.basis})

    def generations(self) -> list[int]:
        """The parameter generations represented in the trail, ascending."""

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
            "generations": self.generations(),
            "breached": sum(1 for entry in self.entries() if entry.breached()),
        }

    @staticmethod
    def _entry(record: Any, voided: dict[int, str] | None = None) -> VerdictEntry:
        payload = record.payload
        detail = payload.get("detail")
        detail = dict(detail) if isinstance(detail, dict) else {}

        def bound(key: str) -> float | None:
            # New records keep bounds at the payload top level; older files
            # wrote them inside detail, so both layouts are read back.
            value = payload.get(key)
            if value is None and key in detail:
                value = detail.pop(key)
            return None if value is None else float(value)

        low = bound("low")
        high = bound("high")
        margin = bound("margin")
        raw_basis = payload.get("basis") or detail.pop("basis", "") or payload.get("name", "")
        raw_generation = payload.get("generation")
        if raw_generation is None:
            raw_generation = detail.pop("generation", 0)
        is_voided = record.sequence in (voided or {})
        return VerdictEntry(
            sequence=record.sequence,
            at=record.at,
            unit=record.unit,
            subject=record.subject,
            name=str(payload.get("name", "")),
            state=str(payload.get("state", "")),
            value=float(payload.get("value", 0.0)),
            actor=record.actor,
            detail=detail,
            generation=int(raw_generation or 0),
            basis=str(raw_basis),
            low=low,
            high=high,
            margin=margin,
            voided=is_voided,
            void_reason=(voided or {}).get(record.sequence, ""),
        )
