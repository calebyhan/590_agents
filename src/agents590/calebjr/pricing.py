"""Provider usage accounting and USD prices per one million tokens."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, cast

MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelPricing:
    input: Decimal
    output: Decimal
    cached_input: Decimal | None
    cache_write_30m: Decimal | None
    cache_write_12h: Decimal | None


def _rates(
    input_rate: str,
    output_rate: str,
    cached: str | None,
    write_30m: str | None,
    write_12h: str | None,
) -> ModelPricing:
    return ModelPricing(
        Decimal(input_rate),
        Decimal(output_rate),
        Decimal(cached) if cached is not None else None,
        Decimal(write_30m) if write_30m is not None else None,
        Decimal(write_12h) if write_12h is not None else None,
    )


# USD per one million tokens, supplied by the UNC gateway pricing schedule.
MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-6-astra": _rates("10", "50", "1", "12.5", "25"),
    "gpt-6.1-sol": _rates("2", "10", "0.1", "2.5", "5"),
    "gpt-6.1-terra": _rates("1", "5", "0.05", "1.25", "2.5"),
    "gpt-6-luna": _rates("0.1", "0.5", "0.01", "0.125", "0.25"),
    "gpt-6-sol": _rates("2", "10", "0.2", "2.5", "5"),
    "gpt-5.6-sol": _rates("4", "20", "0.4", "5", "10"),
    "gpt-5.6-terra": _rates("2", "12", "0.2", "2.5", "5"),
    "gpt-5.6-luna": _rates("0.2", "1.2", "0.02", "0.25", "0.5"),
    "gpt-5.5": _rates("5", "30", "0.5", None, None),
    "gpt-5.5-pro": _rates("30", "180", None, None, None),
    "gpt-5.4": _rates("2.5", "15", "0.25", None, None),
    "gpt-5.4-pro": _rates("30", "180", None, None, None),
}


def _integer(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value).get(name)
    return getattr(value, name, None)


@dataclass
class UsageRecord:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_30m_tokens: int = 0
    cache_write_12h_tokens: int = 0
    reasoning_output_tokens: int = 0
    safety: bool = False
    cache_write_details_available: bool = False

    @classmethod
    def from_response(cls, response: object, *, model: str, safety: bool = False) -> "UsageRecord":
        usage = _field(response, "usage")
        input_details = _field(usage, "input_tokens_details")
        output_details = _field(usage, "output_tokens_details")
        # Accept common gateway aliases while retaining zero when details are absent.
        cached = _integer(_field(input_details, "cached_tokens"))
        write_30m = _integer(_field(input_details, "cache_write_30m_tokens"))
        write_12h = _integer(_field(input_details, "cache_write_12h_tokens"))
        available = any(
            _field(input_details, name) is not None
            for name in ("cache_write_30m_tokens", "cache_write_12h_tokens")
        )
        return cls(
            model=model,
            input_tokens=_integer(_field(usage, "input_tokens")),
            output_tokens=_integer(_field(usage, "output_tokens")),
            cached_input_tokens=cached,
            cache_write_30m_tokens=write_30m,
            cache_write_12h_tokens=write_12h,
            reasoning_output_tokens=_integer(_field(output_details, "reasoning_tokens")),
            safety=safety,
            cache_write_details_available=available,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UsageRecord":
        return cls(
            model=str(value.get("model", "")),
            input_tokens=_integer(value.get("input_tokens")),
            output_tokens=_integer(value.get("output_tokens")),
            cached_input_tokens=_integer(value.get("cached_input_tokens")),
            cache_write_30m_tokens=_integer(value.get("cache_write_30m_tokens")),
            cache_write_12h_tokens=_integer(value.get("cache_write_12h_tokens")),
            reasoning_output_tokens=_integer(value.get("reasoning_output_tokens")),
            safety=bool(value.get("safety", False)),
            cache_write_details_available=bool(value.get("cache_write_details_available", False)),
        )

    def cost(self) -> Decimal | None:
        pricing = MODEL_PRICING.get(self.model)
        if pricing is None:
            return None
        cached = min(self.cached_input_tokens, self.input_tokens)
        write_30m = min(self.cache_write_30m_tokens, self.input_tokens - cached)
        write_12h = min(self.cache_write_12h_tokens, self.input_tokens - cached - write_30m)
        ordinary = max(0, self.input_tokens - cached - write_30m - write_12h)
        categories = (
            (ordinary, pricing.input),
            (self.output_tokens, pricing.output),
            (cached, pricing.cached_input),
            (write_30m, pricing.cache_write_30m),
            (write_12h, pricing.cache_write_12h),
        )
        if any(tokens and rate is None for tokens, rate in categories):
            return None
        return sum(
            (Decimal(tokens) * rate / MILLION for tokens, rate in categories if rate is not None),
            Decimal(0),
        )


@dataclass
class UsageLedger:
    records: list[UsageRecord] = field(default_factory=lambda: list[UsageRecord]())

    def add_response(self, response: object, *, model: str, safety: bool = False) -> None:
        record = UsageRecord.from_response(response, model=model, safety=safety)
        self.records.append(record)

    @property
    def input_tokens(self) -> int:
        return sum(record.input_tokens for record in self.records)

    @property
    def output_tokens(self) -> int:
        return sum(record.output_tokens for record in self.records)

    @property
    def cached_input_tokens(self) -> int:
        return sum(record.cached_input_tokens for record in self.records)

    @property
    def reasoning_output_tokens(self) -> int:
        return sum(record.reasoning_output_tokens for record in self.records)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def requests(self) -> int:
        return len(self.records)

    def cost(self, *, safety: bool | None = None) -> Decimal | None:
        selected = [record for record in self.records if safety is None or record.safety == safety]
        costs = [record.cost() for record in selected]
        if any(cost is None for cost in costs):
            return None
        return sum((cost for cost in costs if cost is not None), Decimal(0))

    def to_list(self) -> list[dict[str, Any]]:
        return [vars(record).copy() for record in self.records]
