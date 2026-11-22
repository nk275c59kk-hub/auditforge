"""Read-only, Gymnasium-style environment for the FinTech audit challenge.

Жанарыс — среда для AI-аудитора.

Supported inputs
----------------
1. A public episode produced by participant 2's ``build_episode``:
       {"episode": {...}, "truth": {...}, "provenance": {...}}
   Only ``episode`` is retained. ``truth`` and ``provenance`` are discarded.
2. A directory written by ``error_generator.write_episode``. Only files in
   ``public/`` are read; ``private/`` is never opened.
3. Participant 1's simple company JSON (company_name, invoices, payments,
   ledger). It is exposed as-is, with a generated manifest and empty FX table.
4. A callable data factory. Factories accepting ``(seed, config)``, ``(seed)``
   or no arguments are supported. This lets ``generate_company(seed)`` be
   passed directly as ``episode_source``.

The environment exposes only five agent tools: list_records, get_record,
get_document, calculator and submit. It has no shell, SQL, filesystem, or
network action. Do not pass private ground-truth data as public episode data.

No third-party packages are required. Python 3.9+.
"""
from __future__ import annotations

import ast
import csv
import inspect
import json
import random
import time
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Callable, Optional, Union


ERROR_TYPES = {
    "duplicate_invoice",
    "payment_allocation",
    "fx_error",
    "period_cutoff",
}
TABLE_ALIASES = {
    "invoice": "invoices",
    "invoices": "invoices",
    "payment": "payments",
    "payments": "payments",
    "ledger": "ledger",
    "record": "ledger",
    "records": "ledger",
    "fx": "fx",
    "exchange_rate": "fx",
    "exchange_rates": "fx",
}
PRIVATE_KEYS = {"truth", "provenance", "ground_truth", "answer_key", "private", "private_path"}
ID_KEYS = (
    "row_id", "entry_id", "record_id", "id", "doc_id", "invoice_id",
    "payment_id", "evidence_id",
)


class AuditorEnv:
    """A small JSON-compatible environment with deterministic episode loading."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        episode_source: Any = None,
        *,
        max_steps: int = 100,
        page_size: int = 50,
        training_mode: bool = True,
        reward_fn: Optional[Callable[[dict, int], float]] = None,
        trajectory_path: Optional[Union[str, Path]] = None,
        time_limit_seconds: Optional[float] = None,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 100:
            raise ValueError("max_steps must be an integer between 1 and 100")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 50:
            raise ValueError("page_size must be an integer between 1 and 50")
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive or None")

        self.episode_source = episode_source
        self.max_steps = max_steps
        self.page_size = page_size
        self.training_mode = bool(training_mode)
        self.__reward_fn = reward_fn
        self.trajectory_path = Path(trajectory_path) if trajectory_path else None
        self.time_limit_seconds = time_limit_seconds

        self.episode_id = ""
        self.__episode: dict[str, Any] = {}
        self.__tables: dict[str, list[dict[str, Any]]] = {}
        self.__known_ids: set[str] = set()
        self.steps = 0
        self.done = True
        self.started_at = 0.0
        self._trajectory_file = None
        self.last_submission: Optional[dict[str, Any]] = None
        self._last_observation: dict[str, Any] = {}
        self._last_seed: Optional[int] = None

    # ----------------------------- lifecycle -----------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        config: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Start a new episode and return ``(observation, info)``.

        ``config`` may contain ``episode_data`` or ``episode_source``. If neither
        is supplied, the source passed to ``AuditorEnv(...)`` is used.
        """
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or None")
        if config is not None and not isinstance(config, dict):
            raise ValueError("config must be a dictionary or None")
        config = dict(config or {})
        raw = config.get("episode_data")
        if raw is None:
            source = config.get("episode_source", self.episode_source)
            if source is None:
                raise ValueError(
                    "No episode source. Pass config={'episode_data': data} or "
                    "set episode_source when creating AuditorEnv."
                )
            raw = self._load_source(source, seed, config)

        public_episode = self._extract_public_episode(raw)
        self.__episode = public_episode
        manifest = self.__episode["manifest"]
        self.episode_id = str(
            config.get("episode_id")
            or manifest.get("episode_id")
            or f"episode-{seed if seed is not None else 0}"
        )
        manifest["episode_id"] = self.episode_id

        self.__tables = {
            "ledger": self._copy_rows(self.__episode.get("ledger", []), "ledger"),
            "invoices": self._copy_rows(self.__episode.get("invoices", []), "invoices"),
            "payments": self._copy_rows(self.__episode.get("payments", []), "payments"),
            "fx": [self._fx_public_row(x) for x in self._copy_rows(self.__episode.get("fx", []), "fx")],
        }
        self.__known_ids = self._collect_known_ids(self.__tables)
        self.steps = 0
        self.done = False
        self.last_submission = None
        self.started_at = time.monotonic()
        self._last_seed = seed
        self._close_trajectory()
        self._open_trajectory()

        public_manifest = self._public_copy(manifest)
        public_manifest.setdefault("n_ledger_rows", len(self.__tables["ledger"]))
        public_manifest.setdefault("n_invoices", len(self.__tables["invoices"]))
        public_manifest.setdefault("n_payments", len(self.__tables["payments"]))
        public_manifest.setdefault(
            "n_documents", len(self.__tables["invoices"]) + len(self.__tables["payments"])
        )
        observation = {
            "episode_id": self.episode_id,
            "task": "Inspect the company's records and submit supported accounting findings with evidence IDs.",
            "manifest": public_manifest,
            "policy": self._public_copy(self.__episode.get("policy", {})),
            "available_actions": [
                "list_records", "get_record", "get_document", "calculator", "submit"
            ],
            "remaining_steps": self.max_steps,
        }
        self._last_observation = observation
        # No seed, truth, provenance, private paths or scoring diagnostics.
        return self._public_copy(observation), {"read_only": True, "seeded": seed is not None}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        """Execute one agent action using the Gymnasium five-value API.

        Every call costs one action, including invalid/unknown actions and submit.
        ``terminated`` is true after submit; ``truncated`` is true at the step/time
        limit. The final action result is retained when the episode is truncated.
        """
        if self.done:
            return (
                {"error": "episode_already_finished", "remaining_steps": 0},
                0.0, True, False, {},
            )

        self.steps += 1
        action = action if isinstance(action, dict) else {}
        action_for_log = self._public_copy(action)
        action_type = action.get("type")
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        try:
            if action_type == "list_records":
                observation = self._list_records(
                    action.get("table"), action.get("cursor"), action.get("limit", self.page_size)
                )
            elif action_type == "get_record":
                observation = self._get_record(action.get("table"), action.get("id"))
            elif action_type == "get_document":
                observation = self._get_document(action.get("doc_id"))
            elif action_type == "calculator":
                observation = self._calculator(action.get("expression"))
            elif action_type == "submit":
                observation, reward, terminated, info = self._submit(action)
            else:
                observation = {"error": "unknown_action"}
        except (ValueError, TypeError, KeyError, InvalidOperation, ArithmeticError) as exc:
            observation = {"error": "invalid_request", "message": str(exc)[:160]}

        if not terminated:
            timed_out = (
                self.time_limit_seconds is not None
                and time.monotonic() - self.started_at >= self.time_limit_seconds
            )
            if self.steps >= self.max_steps or timed_out:
                truncated = True
                self.done = True
                observation = {
                    "message": "time_limit_reached" if timed_out else "step_limit_reached",
                    "last_observation": self._public_copy(observation),
                    "remaining_steps": 0,
                    "final_answer": None,
                }

        observation.setdefault("remaining_steps", max(0, self.max_steps - self.steps))
        self._last_observation = self._public_copy(observation)
        self._record_trajectory(action_for_log, observation, reward, terminated, truncated)
        return self._public_copy(observation), float(reward), terminated, truncated, info

    # ---------------------------- data tools -----------------------------
    def _list_records(self, table: Any, cursor: Any, limit: Any) -> dict:
        canonical = TABLE_ALIASES.get(str(table).lower())
        if canonical is None:
            raise ValueError("Unknown table; use ledger, invoices, payments or fx")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.page_size:
            raise ValueError(f"limit must be an integer from 1 to {self.page_size}")
        offset = self._decode_cursor(cursor)
        rows = self.__tables[canonical]
        if offset > len(rows):
            raise ValueError("Cursor is beyond the end of the table")
        page = rows[offset:offset + limit]
        next_offset = offset + len(page)
        return {
            "table": canonical,
            "records": self._public_copy(page),
            "count": len(page),
            "total": len(rows),
            "next_cursor": f"offset:{next_offset}" if next_offset < len(rows) else None,
        }

    def _get_record(self, table: Any, record_id: Any) -> dict:
        canonical = TABLE_ALIASES.get(str(table).lower())
        if canonical is None:
            raise ValueError("Unknown table")
        if record_id is None:
            raise ValueError("id is required")
        for row in self.__tables[canonical]:
            if any(row.get(key) is not None and str(row[key]) == str(record_id) for key in ID_KEYS):
                return {"table": canonical, "record": self._public_copy(row)}
        return {"error": "record_not_found", "table": canonical}

    def _get_document(self, doc_id: Any) -> dict:
        if doc_id is None:
            raise ValueError("doc_id is required")
        wanted = str(doc_id)
        for table in ("invoices", "payments"):
            for doc in self.__tables[table]:
                aliases = ID_KEYS
                if any(doc.get(key) is not None and str(doc[key]) == wanted for key in aliases):
                    return {
                        "document_type": "invoice" if table == "invoices" else "payment",
                        "document": self._public_copy(doc),
                    }
        return {"error": "document_not_found"}

    # ---------------------------- calculator -----------------------------
    def _calculator(self, expression: Any) -> dict:
        if not isinstance(expression, str) or not expression.strip() or len(expression) > 200:
            raise ValueError("expression must be a non-empty string of at most 200 characters")
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._eval_decimal(tree.body)
        except (SyntaxError, ZeroDivisionError, InvalidOperation, OverflowError, ValueError) as exc:
            raise ValueError("Only safe decimal arithmetic is allowed") from exc
        if not result.is_finite() or len(str(result)) > 100:
            raise ValueError("Result is outside the supported numeric range")
        return {"expression": expression, "result": str(result)}

    @classmethod
    def _eval_decimal(cls, node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            number = cls._eval_decimal(node.operand)
            return number if isinstance(node.op, ast.UAdd) else -number
        if isinstance(node, ast.BinOp):
            left = cls._eval_decimal(node.left)
            right = cls._eval_decimal(node.right)
            with localcontext() as ctx:
                ctx.prec = 50
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
                if isinstance(node.op, ast.Mod):
                    return left % right
                if isinstance(node.op, ast.Pow):
                    if right != right.to_integral_value() or abs(right) > 12:
                        raise ValueError("Exponent must be an integer between -12 and 12")
                    return left ** int(right)
        raise ValueError("Unsupported calculator expression")

    # ------------------------------ submit -------------------------------
    def _submit(self, action: dict) -> tuple[dict, float, bool, dict]:
        payload = action.get("findings")
        error = self._validate_submission(payload)
        if error is None and payload.get("episode_id") != self.episode_id:
            error = "episode_id does not match active episode"
        if error is None:
            error = self._validate_finding_references(payload["findings"])
        if error:
            self.done = True
            # Closed evaluation never exposes reward or validation diagnostics.
            reward = -1.0 if self.training_mode else 0.0
            return (
                {"error": "invalid_submission", "message": "Submission format is invalid", "final_answer": None},
                reward, True, {},
            )

        self.done = True
        self.last_submission = self._public_copy(payload)
        reward = 0.0
        if self.training_mode and self.__reward_fn is not None:
            # The reward callback receives only the agent submission and step count.
            reward = float(self.__reward_fn(self._public_copy(self.last_submission), max(0, self.steps - 1)))
        observation = {
            "message": "submission_received",
            "submitted_findings": len(payload["findings"]),
            "final_answer": self._public_copy(self.last_submission),
        }
        return observation, reward, True, {}

    @staticmethod
    def _validate_submission(payload: Any) -> Optional[str]:
        if not isinstance(payload, dict):
            return "findings must be an object"
        if not isinstance(payload.get("episode_id"), str) or not payload["episode_id"]:
            return "episode_id is required"
        findings = payload.get("findings")
        if not isinstance(findings, list) or len(findings) > 30:
            return "findings must be a list of at most 30"
        for finding in findings:
            if not isinstance(finding, dict):
                return "each finding must be an object"
            if finding.get("type") not in ERROR_TYPES:
                return "unsupported finding type"
            affected = finding.get("affected_ids")
            evidence = finding.get("evidence_ids")
            if not isinstance(affected, list) or not affected or len(affected) > 50:
                return "invalid affected_ids"
            if not all(isinstance(x, (str, int)) and not isinstance(x, bool) for x in affected):
                return "invalid affected_ids"
            if not isinstance(evidence, list) or len(evidence) > 10:
                return "invalid evidence_ids"
            if not all(isinstance(x, (str, int)) and not isinstance(x, bool) for x in evidence):
                return "invalid evidence_ids"
            amount = finding.get("claimed_amount_minor")
            if not isinstance(amount, int) or isinstance(amount, bool):
                return "claimed_amount_minor must be integer"
            if not isinstance(finding.get("currency"), str) or not finding["currency"].strip():
                return "currency is required"
            explanation = finding.get("explanation", "")
            if not isinstance(explanation, str) or len(explanation) > 1000:
                return "explanation too long"
        return None

    def _validate_finding_references(self, findings: list[dict]) -> Optional[str]:
        """Reject references that cannot be found in the public episode."""
        for finding in findings:
            for field in ("affected_ids", "evidence_ids"):
                for value in finding[field]:
                    if str(value) not in self.__known_ids:
                        return "submission references an unknown public evidence ID"
        return None

    # ---------------------------- file loading ---------------------------
    @classmethod
    def _load_source(cls, source: Any, seed: Optional[int], config: dict) -> dict:
        if callable(source):
            return cls._call_factory(source, seed, config)
        if isinstance(source, dict):
            return source
        path = Path(source).expanduser()
        if path.is_dir():
            public = path / "public" if (path / "public").is_dir() else path
            manifest_path = public / "manifest.json"
            policy_path = public / "policy.json"
            return {
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {},
                "policy": json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {},
                "ledger": cls._read_csv(public / "ledger.csv", numeric=("debit_minor", "credit_minor", "debit", "credit")),
                "fx": cls._read_csv(public / "fx.csv"),
                "invoices": cls._read_jsonl(public / "invoices.jsonl"),
                "payments": cls._read_jsonl(public / "payments.jsonl"),
            }
        if not path.is_file():
            raise FileNotFoundError(f"Episode source does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _call_factory(factory: Callable, seed: Optional[int], config: dict) -> Any:
        """Call participant factories with 0, 1 or 2 positional arguments."""
        actual_seed = seed if seed is not None else 0
        try:
            signature = inspect.signature(factory)
            parameters = list(signature.parameters.values())
            accepts_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters)
            positional = [
                p for p in parameters
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if accepts_varargs or len(positional) >= 2:
                call = lambda: factory(actual_seed, config)
            elif len(positional) == 1:
                call = lambda: factory(actual_seed)
            else:
                call = lambda: factory()
        except (TypeError, ValueError):
            call = lambda: factory(actual_seed, config)

        # Some basic generators use module-level random.seed(). Restore the
        # caller's random state so reset does not affect unrelated code.
        state = random.getstate()
        try:
            return call()
        finally:
            random.setstate(state)

    @classmethod
    def _extract_public_episode(cls, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("Episode data must be a JSON object")
        # Participant 2's build_episode returns {episode, truth, provenance}.
        if isinstance(raw.get("episode"), dict):
            raw = raw["episode"]
        elif isinstance(raw.get("public"), dict):
            raw = raw["public"]
        if not isinstance(raw, dict):
            raise ValueError("Public episode must be a JSON object")

        # Whitelist public episode sections. Private wrapper keys are never copied.
        allowed = {"manifest", "policy", "ledger", "invoices", "payments", "fx"}
        clean: dict[str, Any] = {
            key: cls._strip_private(raw.get(key))
            for key in allowed if key in raw
        }
        for key in ("ledger", "invoices", "payments", "fx"):
            value = clean.get(key, [])
            if value is None:
                value = []
            if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
                raise ValueError(f"{key} must be a list of JSON objects")
            clean[key] = value

        # Legacy company.json from participant 1 has no manifest/policy/FX.
        # Keep its original financial rows; add metadata instead of rewriting
        # amounts, IDs, dates or accounting meaning.
        manifest = clean.get("manifest")
        if manifest is None:
            manifest = {}
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        manifest = cls._strip_private(manifest)
        if not manifest.get("base_currency"):
            manifest["base_currency"] = manifest.get("currency") or raw.get("currency") or ""
        if not manifest.get("company_name") and raw.get("company_name"):
            manifest["company_name"] = raw["company_name"]
        manifest.setdefault("schema_version", "legacy-company/1")
        clean["manifest"] = manifest

        policy = clean.get("policy") or {}
        if not isinstance(policy, dict):
            raise ValueError("policy must be a JSON object")
        clean["policy"] = cls._strip_private(policy)
        clean.setdefault("fx", [])
        return clean

    @classmethod
    def _strip_private(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: cls._strip_private(item)
                for key, item in value.items()
                if str(key).lower() not in PRIVATE_KEYS
            }
        if isinstance(value, list):
            return [cls._strip_private(item) for item in value]
        return value

    @classmethod
    def _copy_rows(cls, rows: list[dict], table: str) -> list[dict]:
        return [cls._strip_private(row) for row in rows]

    @classmethod
    def _public_copy(cls, value: Any) -> Any:
        """JSON round-trip provides detached objects and strips private key names."""
        return cls._strip_private(json.loads(json.dumps(value, ensure_ascii=False, default=str)))

    @classmethod
    def _collect_known_ids(cls, tables: dict[str, list[dict]]) -> set[str]:
        known: set[str] = set()
        for rows in tables.values():
            for row in rows:
                for key in ID_KEYS:
                    value = row.get(key)
                    if value is not None:
                        known.add(str(value))
        return known

    @staticmethod
    def _read_csv(path: Path, numeric: tuple[str, ...] = ()) -> list[dict]:
        if not path.exists():
            return []
        output = []
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                item: dict[str, Any] = {}
                for key, value in row.items():
                    if value in (None, "null", ""):
                        item[key] = None
                    elif key in numeric:
                        try:
                            item[key] = int(value)
                        except (TypeError, ValueError):
                            item[key] = value
                    else:
                        item[key] = value
                output.append(item)
        return output

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path.name}, line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Each row in {path.name} must be a JSON object")
            rows.append(row)
        return rows

    @staticmethod
    def _fx_public_row(row: dict) -> dict:
        item = dict(row)
        if not item.get("evidence_id") and all(
            item.get(key) is not None for key in ("date", "source_currency", "target_currency")
        ):
            item["evidence_id"] = (
                f"{item['date']}|{item['source_currency']}|{item['target_currency']}"
            )
        return item

    @staticmethod
    def _decode_cursor(cursor: Any) -> int:
        if cursor is None:
            return 0
        if not isinstance(cursor, str) or not cursor.startswith("offset:"):
            raise ValueError("Invalid cursor")
        value = cursor.split(":", 1)[1]
        try:
            offset = int(value)
        except ValueError as exc:
            raise ValueError("Invalid cursor") from exc
        if offset < 0:
            raise ValueError("Invalid cursor")
        return offset

    # ----------------------------- trajectories --------------------------
    def _open_trajectory(self) -> None:
        if self.trajectory_path is None:
            return
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        self._trajectory_file = self.trajectory_path.open("a", encoding="utf-8")

    def _record_trajectory(
        self,
        action: dict,
        observation: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self._trajectory_file is None:
            return
        record = {
            "episode_id": self.episode_id,
            "step": self.steps,
            "action": action,
            "observation": self._public_copy(observation),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        self._trajectory_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._trajectory_file.flush()

    def _close_trajectory(self) -> None:
        if self._trajectory_file is not None:
            self._trajectory_file.close()
            self._trajectory_file = None

    def close(self) -> None:
        """Close the optional trajectory log."""
        self._close_trajectory()

    def render(self, mode: str = "ansi") -> Optional[str]:
        """Return a compact status view; no accounting or private data is added."""
        if mode not in ("ansi", "human"):
            raise ValueError("mode must be 'ansi' or 'human'")
        status = "finished" if self.done else "active"
        message = f"AuditorEnv({self.episode_id or 'not-reset'}): {status}, steps={self.steps}/{self.max_steps}"
        if mode == "human":
            print(message)
            return None
        return message

    def __enter__(self) -> "AuditorEnv":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
