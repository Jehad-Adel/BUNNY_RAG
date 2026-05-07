from __future__ import annotations

import os
import re
import io
import csv
import time
import logging
from typing import List, Optional, Dict, Literal, Set, Tuple, Any, Union
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
app = FastAPI(title="Bunny RAG")

logger = logging.getLogger(__name__)

MAX_ROWS_DEFAULT = int(os.getenv("MAX_ROWS", "200") or "200")
SCHEMA_CACHE_TTL_SECONDS = int(os.getenv("SCHEMA_CACHE_TTL_SECONDS", "30") or "30")
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5") or "5")
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10") or "10")
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30") or "30")

_LLM_CACHE: Dict[Tuple[str, float], ChatOpenAI] = {}

def get_llm(model: Optional[str] = None, temperature: float = 0.0) -> ChatOpenAI:
    m = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    key = (m, float(temperature))
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = ChatOpenAI(model=m, temperature=temperature)
    return _LLM_CACHE[key]


def log_event(event: str, **kwargs: Any) -> None:
    """Small helper to keep structured logs consistent."""
    logger.info(event, extra={"event": event, **kwargs})


# -----------------------------
# API Models (Pydantic)
# -----------------------------

class QueryRequest(BaseModel):
    text: str = Field(..., description="Natural-language question from the user")


class DebugResponse(BaseModel):
    type: str
    success: bool
    sql: Optional[str] = None
    rows_preview: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    explanation: Optional[str] = None
    message: Optional[str] = None
    strategy: Optional[str] = None
    fallback: Optional[Dict[str, Any]] = None


# =====================================================
# 1) Hardcoded DB schema ("entities")
# =====================================================

class Column(BaseModel):
    name: str


class Table(BaseModel):
    name: str
    columns: List[Column]


class DBSchema(BaseModel):
    dialect: Literal["postgresql"] = "postgresql"
    tables: List[Table]

    def table_names(self) -> Set[str]:
        return {t.name for t in self.tables}

    def all_qualified_columns(self) -> Set[str]:
        out: Set[str] = set()
        for t in self.tables:
            for c in t.columns:
                out.add(f"{t.name}.{c.name}")
        return out

    def get_relationships_hint(self) -> str:
        """Return relationship hints by analyzing actual column names in the schema."""
        t = self.table_names()
        hints: List[str] = ["CRITICAL TABLE RELATIONSHIPS (inferred from schema):"]

        for table in self.tables:
            table_cols = {c.name for c in table.columns}

            if "account_id" in table_cols and "accounts" in t:
                hints.append(f"- {table.name}.account_id → accounts.id")

            if "account_type_id" in table_cols and "account_types" in t:
                hints.append(f"- {table.name}.account_type_id → account_types.id")

            if "plan_id" in table_cols and "plans" in t:
                hints.append(f"- {table.name}.plan_id → plans.id")

            if "price_list_id" in table_cols and "price_lists" in t:
                hints.append(f"- {table.name}.price_list_id → price_lists.id")

            if "subscription_id" in table_cols and "subscriptions" in t:
                hints.append(f"- {table.name}.subscription_id → subscriptions.id")

            if "invoice_id" in table_cols and "invoices" in t:
                hints.append(f"- {table.name}.invoice_id → invoices.id")

            if "contact_id" in table_cols and "contacts" in t:
                hints.append(f"- {table.name}.contact_id → contacts.id")

            if "payment_id" in table_cols and "payments" in t:
                hints.append(f"- {table.name}.payment_id → payments.id")

            if "industry_id" in table_cols and "industries" in t:
                hints.append(f"- {table.name}.industry_id → industries.id")

            if "currency_id" in table_cols and "currencies" in t:
                hints.append(f"- {table.name}.currency_id → currencies.iso_code (currency_id is ISO code like 'EUR', 'USD')")

            if "owner_user_id" in table_cols and "users" in t:
                hints.append(f"- {table.name}.owner_user_id → users.id")

            if "user_id" in table_cols and "users" in t:
                hints.append(f"- {table.name}.user_id → users.id")

            if "deal_id" in table_cols and "deals" in t:
                hints.append(f"- {table.name}.deal_id → deals.id")

            if "deal_stage_id" in table_cols and "deal_stages" in t:
                hints.append(f"- {table.name}.deal_stage_id → deal_stages.id")

            if "quote_id" in table_cols and "quotes" in t:
                hints.append(f"- {table.name}.quote_id → quotes.id")

            if "approval_request_id" in table_cols and "approval_requests" in t:
                hints.append(f"- {table.name}.approval_request_id → approval_requests.id")

            if "approver_id" in table_cols and "approvers" in t:
                hints.append(f"- {table.name}.approver_id → approvers.id")

            if "campaign_id" in table_cols and "campaigns" in t:
                hints.append(f"- {table.name}.campaign_id → campaigns.id")

            if "lead_id" in table_cols and "leads" in t:
                hints.append(f"- {table.name}.lead_id → leads.id")

            if "coupon_id" in table_cols and "coupons" in t:
                hints.append(f"- {table.name}.coupon_id → coupons.id")

        hints.append("\nGENERAL RULES:")
        hints.append("- Use LEFT JOINs to include all rows from the primary table")
        hints.append("- Always use table.column qualified names")
        hints.append("- Use COUNT(DISTINCT ...) when counting unique entities across joins")
        hints.append("- Group by all non-aggregated columns in SELECT")
        hints.append("- For invoice amounts, use invoices.amount column")
        hints.append("- For date filters with 'overdue', compare invoices.due_at with CURRENT_DATE")
        hints.append("- For subscription 'active' state, check subscriptions.state = 'active'")
        hints.append("- CRITICAL: invoices.currency_id contains ISO codes like 'EUR','USD' - join on currencies.iso_code")
        hints.append("- For invoice state filters, valid states include: 'paid', 'posted', 'payment_pending' (NO 'void')")

        return "\n".join(hints) if len(hints) > 1 else "Use standard FK-based joins."


# Define your schema here
schema = DBSchema(
    tables=[
        Table(name="account_balances", columns=[Column(name=c) for c in [
            "id", "account_id", "balance", "created_at", "updated_at", "warren_id", "currency_id"
        ]]),
        Table(name="account_secondary_billing_contacts", columns=[Column(name=c) for c in [
            "id", "warren_id", "account_id", "contact_id"
        ]]),
        Table(name="account_types", columns=[Column(name=c) for c in [
            "id", "warren_id", "name", "created_at", "updated_at", "code"
        ]]),
        Table(name="accounts", columns=[Column(name=c) for c in [
            "id", "warren_id", "account_type_id", "industry_id", "employees", "annual_revenue",
            "name", "billing_street", "billing_city", "billing_state", "billing_zip",
            "billing_country", "shipping_street", "shipping_city", "shipping_state",
            "shipping_zip", "shipping_country", "description", "created_at", "updated_at",
            "phone", "fax", "website", "group_id", "currency_id", "billing_day",
            "net_payment_days", "duns", "owner_user_id", "ip_address", "timezone",
            "billing_contact_id", "uuid", "code", "entity_use_code", "address_validated",
            "demo", "tax_number", "paying_status", "linkedin_url", "invoice_template_id",
            "tax_number_validated", "emails_enabled", "draft_invoices", "entity_id",
            "mrr", "arr", "mur", "new_quote_builder", "disable_dunning",
            "consolidated_billing"
        ]]),
        Table(name="adjustments", columns=[Column(name=c) for c in [
            "id", "warren_id", "invoice_id", "account_id", "amount",
            "description", "created_at", "updated_at", "currency_id", "entity_id"
        ]]),
        Table(name="api_keys", columns=[Column(name=c) for c in [
            "id", "warren_id", "key", "active", "created_at", "updated_at", "entity_id"
        ]]),
        Table(name="audit_events", columns=[Column(name=c) for c in [
            "id", "warren_id", "auditable_type", "auditable_id", "action",
            "audited_changes", "version", "comment", "remote_address",
            "request_uuid", "created_at", "user_id", "entity_id"
        ]]),
        Table(name="bill_runs", columns=[Column(name=c) for c in [
            "id", "warren_id", "accounts_processed", "invoices_generated",
            "amount_invoiced", "last_account_id", "invoice_seq_start",
            "invoice_seq_end", "billing_date", "started_at", "ended_at",
            "state", "demo", "created_at", "updated_at", "runs", "entity_id"
        ]]),
        Table(name="billing_contacts", columns=[Column(name=c) for c in [
            "id", "warren_id", "account_id", "contact_id",
            "created_at", "updated_at"
        ]]),
        Table(name="business_entities", columns=[Column(name=c) for c in [
            "id", "warren_id", "name", "code", "time_zone",
            "logo_url", "created_at", "updated_at"
        ]]),
        Table(name="charges", columns=[Column(name=c) for c in [
            "id", "warren_id", "invoice_id", "price_list_charge_id",
            "quantity", "amount", "description", "created_at",
            "updated_at", "currency_id", "account_id", "tax_amount",
            "tax_rate", "net_amount", "gross_amount",
            "start_date", "end_date", "entity_id"
        ]]),
        Table(name="contacts", columns=[Column(name=c) for c in [
            "id", "warren_id", "first_name", "last_name", "email",
            "phone", "title", "created_at", "updated_at",
            "account_id", "billing_contact",
            "secondary_billing_contact", "shipping_contact",
            "entity_id"
        ]]),
        Table(name="currencies", columns=[Column(name=c) for c in [
            "id", "warren_id", "code", "name", "symbol",
            "created_at", "updated_at"
        ]]),
        Table(name="customer_payment_methods", columns=[Column(name=c) for c in [
            "id", "warren_id", "account_id", "type",
            "token", "created_at", "updated_at", "entity_id"
        ]]),
        Table(name="discounts", columns=[Column(name=c) for c in [
            "id", "warren_id", "name", "code",
            "kind", "value", "created_at",
            "updated_at", "entity_id"
        ]]),
        Table(name="email_events", columns=[Column(name=c) for c in [
            "id", "warren_id", "message_id",
            "event", "payload", "created_at", "updated_at"
        ]]),
        Table(name="entity_users", columns=[Column(name=c) for c in [
            "id", "warren_id", "entity_id",
            "user_id", "role", "created_at", "updated_at"
        ]]),
        Table(name="features", columns=[Column(name=c) for c in [
            "id", "warren_id", "name", "code",
            "created_at", "updated_at", "is_unit",
            "kind", "is_provisioned", "description",
            "product_id", "position", "is_visible"
        ]]),
        Table(name="financial_account_charges", columns=[Column(name=c) for c in [
            "id", "financial_account_id", "price_list_charge_id"
        ]]),
        Table(name="financial_accounts", columns=[Column(name=c) for c in [
            "id", "warren_id", "code", "external_id",
            "plugin_id", "name", "description",
            "active", "source", "created_at",
            "updated_at", "entity_id", "account_type",
            "account_number", "default_account"
        ]]),
        Table(name="frontend_versions", columns=[Column(name=c) for c in [
            "id", "sha", "created_at", "updated_at"
        ]]),
        Table(name="groups", columns=[Column(name=c) for c in [
            "id", "warren_id", "name", "code",
            "created_at", "updated_at"
        ]]),
        Table(name="industries", columns=[Column(name=c) for c in [
            "id", "warren_id", "name", "code",
            "created_at", "updated_at"
        ]]),
        Table(name="invoice_sequences", columns=[Column(name=c) for c in [
            "id", "warren_id", "entity_id",
            "prefix", "number",
            "created_at", "updated_at"
        ]]),
        Table(name="invoice_templates", columns=[Column(name=c) for c in [
            "id", "warren_id", "name", "code",
            "created_at", "updated_at", "entity_id"
        ]]),
        Table(name="invoices", columns=[Column(name=c) for c in [
            "id", "warren_id", "account_id",
            "invoice_number", "state",
            "issue_date", "due_date",
            "subtotal", "tax_total", "total",
            "created_at", "updated_at",
            "currency_id", "billing_contact_id",
            "entity_id", "pdf_url"
        ]]),
        Table(name="line_item_tax_rates", columns=[Column(name=c) for c in [
            "id", "warren_id", "price_list_charge_id",
            "tax_rate_id", "created_at", "updated_at"
        ]]),
        Table(name="payments", columns=[Column(name=c) for c in [
            "id", "warren_id", "account_id",
            "amount", "currency_id",
            "created_at", "updated_at",
            "entity_id", "payment_date",
            "payment_method_id", "state"
        ]]),
        Table(name="price_list_charges", columns=[Column(name=c) for c in [
            "id", "warren_id", "price_list_id",
            "name", "code", "description",
            "unit_price", "created_at", "updated_at",
            "currency_id", "entity_id", "kind"
        ]]),
        Table(name="price_lists", columns=[Column(name=c) for c in [
            "id", "warren_id", "name", "code",
            "created_at", "updated_at",
            "currency_id", "entity_id"
        ]]),
        Table(name="products", columns=[Column(name=c) for c in [
            "id", "warren_id", "name", "code",
            "created_at", "updated_at",
            "entity_id", "description"
        ]]),
        Table(name="tax_rates", columns=[Column(name=c) for c in [
            "id", "warren_id", "name", "code",
            "rate", "created_at", "updated_at", "entity_id"
        ]]),
        Table(name="users", columns=[Column(name=c) for c in [
            "id", "warren_id", "email",
            "encrypted_password", "created_at",
            "updated_at", "reset_password_token",
            "reset_password_sent_at", "remember_created_at",
            "sign_in_count", "current_sign_in_at",
            "last_sign_in_at", "current_sign_in_ip",
            "last_sign_in_ip"
        ]]),
    ]
)


# =====================================================
# 2) Structured SQL model
# =====================================================

class SQLJoin(BaseModel):
    type: Literal["INNER", "LEFT", "RIGHT"] = "INNER"
    table: str
    on: str


class SQLQuery(BaseModel):
    select: List[str]
    from_table: str
    joins: List[SQLJoin] = []
    where: List[str] = []
    group_by: List[str] = []
    having: List[str] = []
    order_by: List[str] = []
    limit: Optional[int] = None

    def to_sql(self) -> str:
        parts: List[str] = ["SELECT " + ", ".join(self.select), "FROM " + self.from_table]

        for j in self.joins:
            parts.append(f"{j.type} JOIN {j.table} ON {j.on}")

        if self.where:
            parts.append("WHERE " + " AND ".join(f"({w})" for w in self.where))

        if self.group_by:
            parts.append("GROUP BY " + ", ".join(self.group_by))

        if self.having:
            parts.append("HAVING " + " AND ".join(f"({h})" for h in self.having))

        if self.order_by:
            parts.append("ORDER BY " + ", ".join(self.order_by))

        if self.limit is not None:
            parts.append("LIMIT " + str(self.limit))

        return "\n".join(parts)


# =====================================================
# 3) Validation
# =====================================================

QUAL_COL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b")


def extract_qualified_refs(expr: str) -> Set[Tuple[str, str]]:
    return set(QUAL_COL_RE.findall(expr))


def validate_query(q: SQLQuery, schema: DBSchema) -> None:
    if q.from_table not in schema.table_names():
        raise ValueError(f"Invalid from_table: {q.from_table}")

    for j in q.joins:
        if j.table not in schema.table_names():
            raise ValueError(f"Invalid join table: {j.table}")

    expressions = (
            q.select + q.where + q.group_by + q.having + q.order_by +
            [j.on for j in q.joins]
    )

    allowed_cols = schema.all_qualified_columns()
    for expr in expressions:
        for t, c in extract_qualified_refs(expr):
            if f"{t}.{c}" not in allowed_cols:
                raise ValueError(f"Invalid column reference {t}.{c}")

    sql_lower = " " + q.to_sql().lower() + " "
    for bad in [" insert ", " update ", " delete ", " drop ", " alter ", " truncate "]:
        if bad in sql_lower:
            raise ValueError("Write operation detected")


def enforce_joins(q: SQLQuery) -> None:
    expressions = (
            q.select + q.where + q.group_by + q.having + q.order_by +
            [j.on for j in q.joins]
    )

    referenced_tables: Set[str] = set()
    for expr in expressions:
        for t, _ in extract_qualified_refs(expr):
            referenced_tables.add(t)

    referenced_tables.discard(q.from_table)
    joined_tables = {j.table for j in q.joins}

    missing = referenced_tables - joined_tables
    if missing:
        raise ValueError(f"Missing JOIN for table(s): {', '.join(sorted(missing))}")


def fix_query_before_validation(q: SQLQuery, schema: DBSchema) -> SQLQuery:
    """
    Fix common LLM mistakes in generated queries BEFORE validation.
    This prevents validation failures for known column name issues.
    """
    # Build a map of table -> columns for quick lookup
    table_cols_map = {}
    for t in schema.tables:
        table_cols_map[t.name] = {c.name for c in t.columns}

    # Known column fixes with alternatives
    column_fixes = [
        # (table, wrong_col, correct_col)
        ("currencies", "id", "iso_code"),  # currencies has no 'id', use iso_code for joins
        ("currencies", "code", "iso_code"),
        ("currencies", "warren_id", "iso_code"),  # For currency joins, use iso_code since currency_id is ISO code
        ("invoices", "due_date", "due_at"),
        ("invoices", "total", "amount"),  # invoices uses 'amount' not 'total'
    ]

    def apply_fixes(text: str) -> str:
        result = text
        for table, wrong_col, correct_col in column_fixes:
            if table in table_cols_map:
                cols = table_cols_map[table]
                # Only fix if wrong column doesn't exist and correct one does
                if wrong_col not in cols and correct_col in cols:
                    result = result.replace(f"{table}.{wrong_col}", f"{table}.{correct_col}")

        # Special fix: if joining currencies on warren_id, use iso_code instead
        # because invoices.currency_id contains ISO codes like 'EUR', 'USD'
        if "currencies" in result and "warren_id" in result:
            result = result.replace("currencies.warren_id", "currencies.iso_code")

        return result

    # Apply fixes to all query parts
    q.select = [apply_fixes(s) for s in q.select]
    q.where = [apply_fixes(w) for w in q.where]
    q.group_by = [apply_fixes(g) for g in q.group_by]
    q.having = [apply_fixes(h) for h in q.having]
    q.order_by = [apply_fixes(o) for o in q.order_by]

    for i, join in enumerate(q.joins):
        q.joins[i].on = apply_fixes(join.on)

    return q


# =====================================================
# 4) LLM → SQL Generation
# =====================================================

def fetch_live_schema(engine: Engine, only_tables: Optional[Set[str]] = None) -> DBSchema:
    """
    Reads tables/columns from PostgresSQL information_schema.
    If only_tables is provided, it filters to those table names.
    """
    filters = """WHERE table_schema = 'public'"""
    if only_tables:
        placeholders = ",".join([f":t{i}" for i, _ in enumerate(sorted(only_tables))])
        filters += f" AND table_name IN ({placeholders})"
    sql = f"""
          SELECT table_name, column_name
          FROM information_schema.columns
          {filters}
          ORDER BY table_name, ordinal_position \
          """
    with engine.connect() as conn:
        params = {f"t{i}": t for i, t in enumerate(sorted(only_tables))} if only_tables else {}
        rows = conn.execute(text(sql), params).mappings().all()

    tables: Dict[str, List[str]] = {}
    for r in rows:
        t = r["table_name"]
        c = r["column_name"]
        tables.setdefault(t, []).append(c)

    return DBSchema(
        tables=[Table(name=t, columns=[Column(name=c) for c in cols]) for t, cols in tables.items()]
    )

_SCHEMA_CACHE: Dict[str, Any] = {}
_SCHEMA_TTL_SECONDS = SCHEMA_CACHE_TTL_SECONDS


def get_live_schema(engine: Engine, only_tables: Optional[Set[str]] = None) -> DBSchema:
    """Cached wrapper around fetch_live_schema."""
    now = time.time()
    key = ",".join(sorted(only_tables)) if only_tables else "__all__"
    cached = _SCHEMA_CACHE.get(key)
    if cached:
        if (now - float(cached["ts"])) < _SCHEMA_TTL_SECONDS:
            return cached["schema"]

    schema = fetch_live_schema(engine, only_tables=only_tables)
    _SCHEMA_CACHE[key] = {"ts": now, "schema": schema}
    return schema


def schema_has_column(s: DBSchema, table: str, column: str) -> bool:
    for t in s.tables:
        if t.name == table:
            return any(c.name == column for c in t.columns)
    return False


def get_compact_schema_repr(schema: DBSchema) -> str:
    """Create a compact text representation of the schema for the LLM prompt."""
    lines = ["Available tables and columns:"]
    for t in schema.tables:
        cols = [c.name for c in t.columns]
        # Truncate column list if too long
        if len(cols) > 15:
            cols_str = ", ".join(cols[:12]) + f", ... ({len(cols)} total columns)"
        else:
            cols_str = ", ".join(cols)
        lines.append(f"- {t.name}: {cols_str}")
    return "\n".join(lines)


def generate_sql(question: str, model: Optional[str] = None) -> Tuple[SQLQuery, DBSchema]:
    """
    Uses OpenAI via LangChain to produce a SQLQuery JSON structure.
    Returns (query, schema_used_for_validation) to keep generation/execution consistent.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

    llm = get_llm(model=model, temperature=0)
    structured = llm.with_structured_output(SQLQuery)

    engine = get_engine()
    full_schema = get_live_schema(engine, only_tables=None)

    question_lower = question.lower()
    keywords = set()

    for table in full_schema.tables:
        table_name_parts = table.name.split('_')
        for part in table_name_parts:
            if part and len(part) > 2:
                if part in question_lower or (part + 's') in question_lower or part.rstrip('s') in question_lower:
                    keywords.add(table.name)
                    break

    core_tables = {'accounts', 'invoices', 'payments', 'subscriptions', 'plans', 'price_lists', 'contacts',
                   'subscription_charges', 'invoice_items', 'products', 'deals', 'disputes', 'quotes',
                   'emails', 'campaigns', 'campaigns_leads', 'users', 'industries', 'currencies',
                   'account_types', 'coupon_applications', 'coupons', 'approval_decisions', 'approvers',
                   'approval_requests', 'quote_user_signers', 'leads', 'deal_stages', 'groups',
                   'bill_runs', 'recurring_revenue', 'revenues'}
    relevant_tables = keywords.union(core_tables)

    additional_tables = set()
    for table in full_schema.tables:
        if table.name in relevant_tables:
            for col in table.columns:
                if col.name.endswith('_id') and col.name != 'id':
                    potential_table = col.name[:-3]
                    if potential_table + 's' in [t.name for t in full_schema.tables]:
                        additional_tables.add(potential_table + 's')
                    elif potential_table in [t.name for t in full_schema.tables]:
                        additional_tables.add(potential_table)

    relevant_tables = relevant_tables.union(additional_tables)

    if len(relevant_tables) > 0 and len(relevant_tables) < len(full_schema.tables):
        filtered_tables = [t for t in full_schema.tables if t.name in relevant_tables]
        live_schema = DBSchema(tables=filtered_tables)
        logger.info(f"Filtered schema: {len(filtered_tables)}/{len(full_schema.tables)} tables: {sorted([t.name for t in filtered_tables])}")
    else:
        live_schema = full_schema
        logger.info(f"Using full schema: {len(full_schema.tables)} tables")

    # Create compact schema representation for prompt
    compact_schema = get_compact_schema_repr(live_schema)

    prompt = f"""Generate a PostgreSQL SELECT query for: "{question}"

SCHEMA:
{compact_schema}

RULES:
- Table-qualify all columns (table.column)
- Use proper JOINs (no implicit joins)
- For aggregates: use SUM/COUNT/AVG with COALESCE() and GROUP BY
- For counts: use COUNT(DISTINCT ...) when counting unique entities

CRITICAL COLUMN MAPPINGS:
- accounts.mrr = monthly recurring revenue (numeric column, can filter with > or <)
- accounts.arr = annual recurring revenue
- invoices.amount = invoice total amount
- invoices.due_at = invoice due date (NOT due_date)
- invoices.state = invoice status ('paid', 'posted', etc. - NO 'void')
- invoices.currency_id = ISO currency code like 'EUR', 'GBP' (VARCHAR)
- invoices.created_at = invoice creation date
- deals.amount = deal value
- payments.amount = payment amount
- subscriptions.state = subscription status ('active', 'cancelled', etc.)
- subscription_charges.amount = charge amount per subscription
- currencies.iso_code = currency ISO code (join with invoices.currency_id)
- bill_runs.billing_date = billing date for revenue trends
- bill_runs.amount_invoiced = total invoiced amount in bill run

QUERY EXAMPLES:
- Accounts with MRR > 1000: SELECT accounts.id, accounts.name, accounts.mrr FROM accounts WHERE accounts.mrr > 1000
- Monthly revenue trend: SELECT TO_CHAR(invoices.created_at, 'YYYY-MM') AS month, SUM(invoices.amount) AS revenue FROM invoices WHERE EXTRACT(YEAR FROM invoices.created_at) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY TO_CHAR(invoices.created_at, 'YYYY-MM') ORDER BY month
- Account with invoice totals: SELECT accounts.id, accounts.name, COALESCE(SUM(invoices.amount), 0) AS total FROM accounts LEFT JOIN invoices ON accounts.id = invoices.account_id GROUP BY accounts.id, accounts.name
- Account with multiple counts: SELECT accounts.id, accounts.name, COUNT(DISTINCT invoices.id) AS invoice_count, COUNT(DISTINCT payments.id) AS payment_count FROM accounts LEFT JOIN invoices ON accounts.id = invoices.account_id LEFT JOIN payments ON accounts.id = payments.account_id GROUP BY accounts.id, accounts.name

{live_schema.get_relationships_hint()}

Return ONLY the SQLQuery JSON structure.
""".strip()

    try:
        q = structured.invoke(prompt)
        logger.info(f"LLM generated query: from_table={q.from_table}, select={q.select[:2]}...")
    except Exception as e:
        logger.error(f"LLM structured output failed for question: {question}. Error: {e}")
        raise ValueError(f"LLM failed to generate valid SQL structure: {e}")

    # Fix common LLM mistakes before validation
    q = fix_query_before_validation(q, live_schema)

    validate_query(q, live_schema)
    enforce_joins(q)
    return q, live_schema


def generate_sql_with_retry(question: str, model: Optional[str] = None, max_attempts: int = 1) -> Tuple[SQLQuery, DBSchema]:
    """
    Try generating SQL with filtered schema first.
    If it fails validation, retry once with full schema.
    This keeps API calls low while preventing false failures.
    """
    last_err: Optional[Exception] = None

    # First attempt: with filtered schema
    try:
        return generate_sql(question, model=model)
    except Exception as e:
        last_err = e
        msg = str(e).lower()
        if "rate limit" in msg or "429" in msg:
            raise

        logger.warning(f"Filtered schema generation failed: {e}. Retrying with full schema...")

    # Second attempt: force full schema (bypass filtering)
    try:
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        llm = get_llm(model=model, temperature=0)
        structured = llm.with_structured_output(SQLQuery)

        engine = get_engine()
        full_schema = get_live_schema(engine, only_tables=None)

        # Create compact schema representation
        compact_schema = get_compact_schema_repr(full_schema)

        prompt = f"""Generate a PostgreSQL SELECT query for: "{question}"

SCHEMA:
{compact_schema}

RULES:
- Table-qualify all columns (table.column)
- Use proper JOINs (no implicit joins)
- For aggregates: use SUM/COUNT/AVG with COALESCE() and GROUP BY
- For counts: use COUNT(DISTINCT ...) when counting unique entities

CRITICAL COLUMN MAPPINGS:
- accounts.mrr = monthly recurring revenue (numeric column, can filter with > or <)
- accounts.arr = annual recurring revenue
- invoices.amount = invoice total amount
- invoices.due_at = invoice due date (NOT due_date)
- invoices.state = invoice status ('paid', 'posted', etc. - NO 'void')
- invoices.currency_id = ISO currency code like 'EUR', 'GBP' (VARCHAR)
- invoices.created_at = invoice creation date
- deals.amount = deal value
- payments.amount = payment amount
- subscriptions.state = subscription status ('active', 'cancelled', etc.)
- subscription_charges.amount = charge amount per subscription
- currencies.iso_code = currency ISO code (join with invoices.currency_id)
- bill_runs.billing_date = billing date for revenue trends
- bill_runs.amount_invoiced = total invoiced amount in bill run

QUERY EXAMPLES:
- Accounts with MRR > 1000: SELECT accounts.id, accounts.name, accounts.mrr FROM accounts WHERE accounts.mrr > 1000
- Monthly revenue trend: SELECT TO_CHAR(invoices.created_at, 'YYYY-MM') AS month, SUM(invoices.amount) AS revenue FROM invoices WHERE EXTRACT(YEAR FROM invoices.created_at) = EXTRACT(YEAR FROM CURRENT_DATE) GROUP BY TO_CHAR(invoices.created_at, 'YYYY-MM') ORDER BY month
- Account with invoice totals: SELECT accounts.id, accounts.name, COALESCE(SUM(invoices.amount), 0) AS total FROM accounts LEFT JOIN invoices ON accounts.id = invoices.account_id GROUP BY accounts.id, accounts.name
- Account with multiple counts: SELECT accounts.id, accounts.name, COUNT(DISTINCT invoices.id) AS invoice_count, COUNT(DISTINCT payments.id) AS payment_count FROM accounts LEFT JOIN invoices ON accounts.id = invoices.account_id LEFT JOIN payments ON accounts.id = payments.account_id GROUP BY accounts.id, accounts.name

{full_schema.get_relationships_hint()}

Return ONLY the SQLQuery JSON structure.
""".strip()

        q = structured.invoke(prompt)

        # Fix common LLM mistakes before validation
        q = fix_query_before_validation(q, full_schema)

        validate_query(q, full_schema)
        enforce_joins(q)
        logger.info("Full schema generation succeeded")
        return q, full_schema

    except Exception as e2:
        logger.error(f"Full schema generation also failed: {e2}")
        raise last_err if last_err else e2


# =====================================================
# 5) Database execution
# =====================================================

_engine: Optional[Engine] = None


def build_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:"
        f"{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:"
        f"{os.getenv('DB_PORT')}/"
        f"{os.getenv('DB_NAME')}"
    )
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_timeout=DB_POOL_TIMEOUT,
    )
    logger.info(
        "DB engine configured",
        extra={
            "pool_size": DB_POOL_SIZE,
            "max_overflow": DB_MAX_OVERFLOW,
            "pool_timeout": DB_POOL_TIMEOUT,
        },
    )
    return engine


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = build_engine()
    return _engine


def get_max_rows() -> int:
    try:
        return MAX_ROWS_DEFAULT
    except ValueError:
        return 200


def normalize_aggregates(q: SQLQuery) -> SQLQuery:
    """
    Make aggregate queries more robust:
    - Wrap SUM(x) with COALESCE(SUM(x), 0) to avoid NULL totals when no rows match.
    - DO NOT add HAVING clauses automatically as they can reference aliases incorrectly
    """
    sum_pattern = re.compile(r"\bSUM\(\s*([^)]+)\s*\)", re.IGNORECASE)

    new_select: List[str] = []
    for sel in q.select:
        if "SUM(" in sel.upper() and "COALESCE(SUM(" not in sel.upper():
            sel = sum_pattern.sub(r"COALESCE(SUM(\1), 0)", sel)
        new_select.append(sel)

    q.select = new_select

    return q


def run_sqlquery(engine: Engine, q: SQLQuery, schema_to_use: DBSchema, max_rows: int = 200) -> List[Dict[str, Any]]:
    # Validate on the SAME schema used for generation
    validate_query(q, schema_to_use)
    enforce_joins(q)

    q.limit = min(max_rows, q.limit or max_rows)

    sql = q.to_sql()

    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        conn.execute(text("SET statement_timeout = '30s'"))
        result = conn.execute(text(sql))
        return [dict(r) for r in result.mappings().all()]


def rows_to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    """Convert rows to CSV bytes, handling empty results."""
    buf = io.StringIO()
    if not rows:
        return b"message\nNo rows returned for this query\n"

    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def generate_fallback_explanation_safe(question: str, error: str) -> Dict[str, Any]:
    # Safe, non-LLM fallback payload (so debug endpoint never crashes)
    return {
        "question": question,
        "error": error,
        "note": "Fallback returned because SQL generation/execution failed."
    }


# =====================================================
# 6) Advice-mode (Generalized)
# =====================================================

def run_readonly_sql(engine: Engine, sql: str, max_rows: int) -> List[Dict[str, Any]]:
    """Execute SQL with safety limits and read-only transaction."""
    sql_clean = sql.strip().rstrip(";")
    enforced_limit = int(max_rows)
    if " limit " not in sql_clean.lower():
        sql_clean = f"{sql_clean}\nLIMIT {enforced_limit}"
    else:
        sql_clean = re.sub(r"(?i)limit\s+(\d+)", lambda m: f"LIMIT {min(int(m.group(1)), enforced_limit)}", sql_clean)

    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        conn.execute(text("SET statement_timeout = '30s'"))
        res = conn.execute(text(sql_clean)).mappings().all()
    return [dict(r) for r in res]


def generate_advice_answer(question: str, data_summary: Dict[str, Any]) -> str:
    """
    Use OpenAI to produce a strategic answer based on provided data.
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = get_llm(model=model, temperature=0)

    prompt = f"""
You are a senior business analyst. Based on the data provided, answer the user's strategic question.

User Question: {question}

Data Summary:
{data_summary}

Instructions:
- Analyze the data and provide 3-5 actionable insights
- Be specific and reference actual data points
- If data is insufficient, explain what's missing
- Keep it concise and professional
- Use bullet points for clarity
- Do NOT fabricate data

Respond with a clear, actionable analysis.
"""
    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))


def generate_nl_answer(question: str, sql: str, rows: List[Dict[str, Any]]) -> str:
    """
    Uses OpenAI to summarize the SQL result in natural language.
    Optimized with shorter prompts and row limits.
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = get_llm(model=model, temperature=0)

    preview = rows[:10]

    prompt = f"""Answer: "{question}"

Data (top {len(preview)} of {len(rows)} rows):
{preview}

Return a clear, 2-3 sentence answer highlighting key findings. If it's a ranking, mention top 3-5 items."""

    try:
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp))
    except Exception as e:
        logger.warning(f"NL answer generation failed: {e}")
        return f"Query returned {len(rows)} rows. See results below."


def repair_query_for_postgres(q: SQLQuery, err_msg: str, schema: DBSchema, question: str = "") -> SQLQuery:
    """Deterministic fixes for common Postgres execution errors."""
    em = (err_msg or "").lower()

    # Fix due_date -> due_at (database uses due_at, not due_date)
    if "due_date" in em or 'column "due_date" does not exist' in em:
        for i, where_clause in enumerate(q.where):
            q.where[i] = where_clause.replace("invoices.due_date", "invoices.due_at")
        for i, sel in enumerate(q.select):
            q.select[i] = sel.replace("invoices.due_date", "invoices.due_at")
        for i, order in enumerate(q.order_by):
            q.order_by[i] = order.replace("invoices.due_date", "invoices.due_at")

    # Fix currencies joins - currency_id contains ISO codes, not numeric IDs
    # So we need to join on currencies.iso_code, not currencies.warren_id or currencies.id
    if "currencies" in em:
        for i, join in enumerate(q.joins):
            if "currencies" in join.on:
                q.joins[i].on = join.on.replace("currencies.id", "currencies.iso_code")
                q.joins[i].on = q.joins[i].on.replace("currencies.warren_id", "currencies.iso_code")
        for i, sel in enumerate(q.select):
            q.select[i] = sel.replace("currencies.id", "currencies.iso_code")

    # Fix total -> amount for invoices if total doesn't exist
    if 'column "total" does not exist' in em and 'invoices' in em:
        for i, sel in enumerate(q.select):
            q.select[i] = sel.replace("invoices.total", "invoices.amount")
        for i, where_clause in enumerate(q.where):
            q.where[i] = where_clause.replace("invoices.total", "invoices.amount")
        for i, order in enumerate(q.order_by):
            q.order_by[i] = order.replace("invoices.total", "invoices.amount")

    # Fix invoice state - 'void' is not a valid value, remove it from NOT IN clauses
    if 'void' in em or 'invalid input value for enum' in em:
        for i, where_clause in enumerate(q.where):
            q.where[i] = where_clause.replace("NOT IN ('paid', 'void')", "!= 'paid'")
            q.where[i] = q.where[i].replace("('paid', 'void')", "('paid')")

    # Fix subscription state for active check
    if "subscriptions" in question.lower() and "active" in question.lower():
        for i, where_clause in enumerate(q.where):
            if "state = 'active'" in where_clause or "state IN" in where_clause:
                q.where[i] = where_clause.replace("state IN ('active', 'posted', 'open')", "state = 'active'")

    if "this quarter" in question.lower() and "due_at" not in " ".join(q.where).lower():
        q.where.append(
            "invoices.due_at >= date_trunc('quarter', CURRENT_DATE) "
            "AND invoices.due_at < date_trunc('quarter', CURRENT_DATE) + interval '3 months'"
        )

    return q


# =====================================================
# 7) Main orchestration
# =====================================================

def is_conversational_question(question: str) -> bool:
    """Detect if question is conversational vs data query."""
    q_lower = question.lower().strip()

    # Remove punctuation for better matching
    q_clean = re.sub(r'[^\w\s]', '', q_lower)
    words = q_clean.split()

    # Exact match patterns (high confidence conversational)
    exact_conversational = [
        "hello", "hi", "hey", "greetings", "howdy",
        "bye", "goodbye", "thanks", "thank you", "cheers"
    ]

    # If the entire message is just a greeting word
    if q_clean in exact_conversational:
        return True

    # Phrase patterns that indicate conversational intent
    conversational_patterns = [
        "hello", "hi there", "hey there", "how are you", "who are you",
        "what are you", "what is your name", "whats your name",
        "help me", "can you help", "what can you do", "how do you work",
        "thank you", "thanks", "bye", "goodbye", "good morning",
        "good afternoon", "good evening", "nice to meet you",
        "how's it going", "hows it going", "what's up", "whats up"
    ]

    # Check for conversational patterns (allow longer questions if they match patterns)
    for pattern in conversational_patterns:
        if pattern in q_lower:
            return True

    # Short questions starting with greeting words are likely conversational
    if len(words) <= 8 and words and words[0] in ["hello", "hi", "hey", "thanks", "bye"]:
        # But not if they contain data-related keywords
        data_keywords = ["show", "list", "count", "total", "sum", "average", "how many",
                         "which", "what is the", "find", "get", "display", "report"]
        if not any(kw in q_lower for kw in data_keywords):
            return True

    return False


def handle_conversational_question(question: str) -> str:
    """Generate conversational response for non-data questions."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = get_llm(model=model, temperature=0.7)

    prompt = f"""You are a helpful database assistant. The user asked: "{question}"

This is not a data query question. Respond naturally and helpfully.

If they're greeting you: Be friendly and introduce yourself as a database assistant.
If they're asking what you can do: Explain you can answer questions about their database (accounts, invoices, payments, subscriptions, plans, etc.)
If they're asking for help: Suggest example questions like "list plans by subscribers" or "show top accounts"
If they're thanking you: Respond politely.

Keep response brief (2-3 sentences)."""

    try:
        logger.info(f"Generating conversational response for: {question}")
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", str(resp))
        logger.info(f"Conversational response generated: {content[:100]}...")
        return content if content else "I'm a database assistant. How can I help you explore your data?"
    except Exception as e:
        logger.error(f"Conversational response failed: {e}", exc_info=True)
        return "I'm a database assistant. I can help you query your data! Try asking questions like 'list plans by subscribers' or 'show top 10 accounts'."


def handle_question(question: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """Main orchestration: detect question type, generate SQL or conversational response."""
    if max_rows is None:
        max_rows = get_max_rows()

    if is_conversational_question(question):
        logger.info(f"Detected conversational question: {question}")
        answer = handle_conversational_question(question)
        logger.info(f"Returning conversational response: {answer[:100]}...")
        return {
            "type": "conversational",
            "success": True,
            "sql": None,
            "rows": [],
            "row_count": 0,
            "explanation": answer,
            "message": None,
            "fallback": None,
        }

    engine = get_engine()

    last_err: Optional[Exception] = None
    q: Optional[SQLQuery] = None
    schema_used: Optional[DBSchema] = None

    try:
        q, schema_used = generate_sql_with_retry(question)
        q = normalize_aggregates(q)
        sql = q.to_sql()

        rows = run_sqlquery(engine, q, schema_to_use=schema_used, max_rows=max_rows)

        if len(rows) > 0:
            answer = generate_nl_answer(question, sql, rows)
        else:
            answer = f"No matching records found for your query. The query executed successfully but returned 0 rows. You may want to check if:\n- The data exists in the database\n- Your filters are not too restrictive\n- The table relationships are correct"

        return {
            "type": "sql",
            "success": True,
            "sql": sql,
            "rows": rows,
            "row_count": len(rows),
            "explanation": answer,
            "message": None,
            "fallback": None,
        }

    except Exception as e:
        last_err = e
        logger.error(f"Query generation/execution failed: {e}", exc_info=True)

        if q is not None and schema_used is not None:
            try:
                q = repair_query_for_postgres(q, str(e), schema_used, question)
                q = normalize_aggregates(q)
                sql = q.to_sql()
                rows = run_sqlquery(engine, q, schema_to_use=schema_used, max_rows=max_rows)

                if len(rows) > 0:
                    answer = generate_nl_answer(question, sql, rows)
                else:
                    answer = "Query was repaired but returned no data. The query structure is valid but no matching records exist."

                return {
                    "type": "sql",
                    "success": True,
                    "sql": sql,
                    "rows": rows,
                    "row_count": len(rows),
                    "explanation": answer,
                    "message": None,
                    "fallback": None,
                }
            except Exception as repair_error:
                logger.error(f"Query repair failed: {repair_error}", exc_info=True)
                last_err = repair_error

    error_msg = str(last_err) if last_err else "Unknown error"
    logger.error(f"All query attempts failed for: {question}. Error: {error_msg}")

    return {
        "type": "fallback",
        "success": False,
        "sql": None,
        "rows": [],
        "row_count": 0,
        "explanation": None,
        "message": "Unable to answer from the database with the current schema/rules.",
        "strategy": None,
        "fallback": generate_fallback_explanation_safe(question, error_msg),
    }


# =====================================================
# 8) API Endpoints
# =====================================================

@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        api_key = os.getenv("OPENAI_API_KEY")
        has_openai = bool(api_key)

        return {
            "status": "healthy",
            "database": "connected",
            "openai": "configured" if has_openai else "missing",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "database": "disconnected",
            "error": "Database connection failed",
            "timestamp": datetime.utcnow().isoformat(),
        }


@app.post("/query.csv")
def query_csv(req: QueryRequest):
    """CSV export endpoint - always returns CSV (even if empty)."""
    try:
        result = handle_question(req.text)
        rows = result.get("rows") or []

        csv_bytes = rows_to_csv_bytes(rows)
        headers = {"Content-Disposition": 'attachment; filename="query_results.csv"'}
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv; charset=utf-8",
            headers=headers,
        )

    except Exception as e:
        logger.error(f"CSV export error: {e}", exc_info=True)
        csv_bytes = "message\nUnable to generate CSV. Please try a different query.\n".encode("utf-8")
        headers = {"Content-Disposition": 'attachment; filename="query_results.csv"'}
        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv; charset=utf-8",
            headers=headers,
        )


@app.post("/query.debug", response_model=DebugResponse)
def query_debug(req: QueryRequest):
    """Debug endpoint - always returns valid JSON."""
    try:
        result = handle_question(req.text)

        response_data: Dict[str, Any] = {
            "type": result.get("type", "fallback"),
            "success": bool(result.get("success", False)),
        }

        rtype = response_data["type"]

        if rtype == "sql":
            response_data.update({
                "sql": result.get("sql"),
                "rows_preview": (result.get("rows") or [])[:50],
                "row_count": result.get("row_count"),
                "explanation": result.get("explanation"),
            })
        else:
            explanation = result.get("explanation")
            message = result.get("message") or explanation or "The requested data is not available in the database."
            response_data.update({
                "sql": result.get("sql"),
                "message": message,
                "explanation": explanation,
                "fallback": result.get("fallback"),
                "rows_preview": (result.get("rows") or [])[:50] if result.get("rows") is not None else None,
                "row_count": result.get("row_count"),
            })

        return DebugResponse(**response_data)

    except Exception as e:
        logger.error(f"Debug endpoint error: {e}", exc_info=True)
        return DebugResponse(
            type="fallback",
            success=False,
            message="The requested data is not available in the database.",
            fallback=generate_fallback_explanation_safe(req.text, str(e)),
        )


# =====================================================
# 9) Frontend
# =====================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return r"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Bunny RAG</title>

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

    <style>
      :root{
        --bg: #f7f7f8;
        --panel: #ffffff;
        --text: #0f172a;
        --muted: #64748b;
        --border: rgba(15, 23, 42, .10);
        --border-strong: rgba(15, 23, 42, .18);
        --shadow: 0 1px 2px rgba(0,0,0,.05), 0 8px 30px rgba(0,0,0,.06);
        --radius: 16px;

        --accent: #0f172a;          /* mostly monochrome */
        --accent-soft: rgba(15,23,42,.08);

        --danger: #b91c1c;
        --danger-bg: #fff1f2;
        --danger-br: #fecdd3;
      }

      *{ box-sizing:border-box; }
      html, body { height: 100%; }
      body{
        margin:0;
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        background: var(--bg);
        color: var(--text);
      }

      /* Top bar */
      .topbar{
        position: sticky;
        top: 0;
        z-index: 10;
        background: rgba(247,247,248,.85);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border);
      }
      .topbar-inner{
        width: 100%;
        padding: 14px 18px;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 12px;
      }
      .brand{
        display:flex;
        align-items:center;
        gap:10px;
        font-weight:600;
        letter-spacing:-.01em;
      }
      .logo{
        width: 28px;
        height: 28px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(15,23,42,.92), rgba(15,23,42,.55));
        box-shadow: 0 1px 2px rgba(0,0,0,.12);
      }
      .status-pill{
        display:inline-flex;
        align-items:center;
        gap: 8px;
        font-size: 12px;
        color: var(--muted);
        border: 1px solid var(--border);
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,.7);
      }
      .dot{
        width: 8px; height: 8px; border-radius: 999px;
        background: #22c55e;
        box-shadow: 0 0 0 3px rgba(34,197,94,.15);
      }

      /* Full-screen app frame */
      .wrap{
        width: 100%;
        height: calc(100vh - 57px); /* topbar height */
        padding: 0;
        margin: 0;
      }

      .panel{
        height: 100%;
        width: 100%;
        background: var(--panel);
        border-top: 0;
        border-left: 0;
        border-right: 0;
        border-bottom: 0;
        border-radius: 0;
        box-shadow: none;
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }

      .panel-header{
        padding: 18px 18px 10px;
        border-bottom: 1px solid rgba(15, 23, 42, .06);
        background: rgba(255,255,255,.9);
      }
      h1{
        margin:0;
        font-size: 20px;
        letter-spacing:-.02em;
      }
      .sub{
        margin-top: 6px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
      }

      .content{
        flex: 1;
        min-height: 0;
        padding: 18px;
        display:grid;
        grid-template-columns: 1.1fr .9fr;
        gap: 14px;
        align-items: stretch;
      }
      @media (max-width: 920px){
        .content{ grid-template-columns: 1fr; }
      }

      /* Left input card */
      .card{
        border: 1px solid var(--border);
        border-radius: 14px;
        background: #fff;
        overflow:hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
        min-height: 0;
      }
      .card-top{
        padding: 14px 14px 10px;
        border-bottom: 1px solid rgba(15, 23, 42, .06);
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:10px;
      }
      label{
        font-weight:600;
        font-size: 13px;
        letter-spacing:-.01em;
      }
      .hint{
        color: var(--muted);
        font-size: 12px;
        margin-top: 4px;
        line-height: 1.35;
      }

      .right-meta{
        color: var(--muted);
        font-size: 12px;
        display:flex;
        gap: 8px;
        align-items:center;
        flex-wrap:wrap;
        justify-content:flex-end;
      }
      .badge{
        display:inline-flex;
        align-items:center;
        gap:6px;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(15,23,42,.03);
      }

      .textarea-wrap{
        position: relative;
        flex: 1;
        min-height: 0;
      }
      textarea{
        width:100%;
        height: 100%;
        min-height: 0;
        border: 0;
        outline:none;
        resize: none; /* looks more app-like */
        padding: 12px 14px 14px;
        font-size: 14px;
        line-height: 1.45;
        background: transparent;
      }
      .kbd{
        position:absolute;
        right: 12px;
        bottom: 12px;
        display:flex;
        gap:6px;
        color: var(--muted);
        font-size: 11px;
        user-select:none;
      }
      .kbd span{
        border: 1px solid var(--border);
        background: rgba(15,23,42,.03);
        padding: 3px 6px;
        border-radius: 7px;
      }

      .actions{
        padding: 12px 14px 14px;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 10px;
        border-top: 1px solid rgba(15, 23, 42, .06);
      }
      .btn{
        display:inline-flex;
        align-items:center;
        justify-content:center;
        gap:8px;
        border-radius: 12px;
        padding: 10px 12px;
        font-weight:600;
        font-size: 13px;
        border: 1px solid var(--border);
        background: rgba(15,23,42,.03);
        cursor:pointer;
      }
      .btn:hover{ background: rgba(15,23,42,.05); }
      .btn:disabled{ opacity:.6; cursor:not-allowed; }

      .btn-primary{
        background: var(--accent);
        color: white;
        border-color: rgba(0,0,0,.15);
      }
      .btn-primary:hover{ filter: brightness(1.03); }

      .muted{ color: var(--muted); }
      .small{ font-size: 12px; }
      .big-output{ font-size: 14px; font-weight:500; }
      code{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }

      /* Right output card */
      .output{
        border: 1px solid var(--border);
        border-radius: 14px;
        background: #fff;
        overflow:hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
        min-height: 0;
      }
      .output-head{
        padding: 14px;
        border-bottom: 1px solid rgba(15, 23, 42, .06);
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 10px;
      }
      .output-title{
        font-weight:600;
        font-size: 13px;
        letter-spacing:-.01em;
      }
      .output-body{
        padding: 14px;
        flex: 1;
        min-height: 0;
        overflow: auto;
      }

      .section-title{
        margin: 10px 0 8px;
        font-weight: 600;
        font-size: 12px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: .08em;
      }
      .divider{ height: 1px; background: rgba(15,23,42,.06); margin: 12px 0; }

      pre{
        margin: 0;
        background: rgba(15,23,42,.03);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px;
        overflow:auto;
        font-size: 12.5px;
        line-height: 1.45;
      }

      table{
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
      }
      th, td{
        padding: 9px 10px;
        text-align:left;
        font-size: 12.5px;
        border-bottom: 1px solid rgba(15, 23, 42, .07);
        vertical-align: top;
        white-space: nowrap;
      }
      td{ white-space: normal; }
      th{
        background: rgba(15,23,42,.03);
        font-weight: 600;
      }
      tr:last-child td{ border-bottom: 0; }

      .err{
        color: var(--danger);
        background: var(--danger-bg);
        border: 1px solid var(--danger-br);
        padding: 10px 12px;
        border-radius: 12px;
        white-space: pre-wrap;
        font-size: 13px;
      }

      .dlrow{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 10px;
        flex-wrap: wrap;
      }
    </style>
  </head>

  <body>
    <div class="topbar">
      <div class="topbar-inner">
        <div class="brand">
          <div class="logo" aria-hidden="true"></div>
          <div>Bunny RAG</div>
        </div>
        <div class="status-pill">
          <span class="dot" id="dot"></span>
          <span id="status">Online</span>
        </div>
      </div>
    </div>

    <div class="wrap">
      <div class="panel">
        <div class="panel-header">
          <h1>Ask your database in plain English</h1>
          <div class="sub">
            Ctrl+Enter (or ⌘+Enter) to run. Server enforces <b>SELECT-only</b>, allowlisted schema, and read-only transactions.
          </div>
        </div>

        <div class="content">
          <!-- Left: input -->
          <div class="card">
            <div class="card-top">
              <div>
                <label for="q">Question</label>
                <div class="hint">Example: "List 20 accounts with their account type name".</div>
              </div>
              <div class="right-meta">
                <span class="badge">SELECT-only</span>
                <span class="badge">Allowlisted</span>
                <span class="badge">Read-only</span>
              </div>
            </div>

            <div class="textarea-wrap">
              <textarea id="q" spellcheck="false">List 20 accounts with their account type name</textarea>
              <div class="kbd" aria-hidden="true">
                <span>Ctrl</span><span>Enter</span>
              </div>
            </div>

            <div class="actions">
              <button id="runBtn" class="btn btn-primary">Run</button>
              <div class="muted small" id="meta">Ready.</div>
            </div>
          </div>

          <!-- Right: output -->
          <div class="output" id="out">
            <div class="output-head">
              <div class="output-title">Result</div>
              <button id="dlBtn" class="btn" style="display:none;">Download CSV</button>
            </div>

            <div class="output-body">
              <div id="error" style="display:none;"></div>

              <div id="empty" class="muted small">
                Run a question to see generated SQL and a preview.
              </div>

              <div id="result" style="display:none;">
                <div class="dlrow muted small" id="rowInfo"></div>

                <div class="divider"></div>

                <div class="section-title">Answer</div>
                <div id="answer" class="muted big-output" style="white-space:pre-wrap;"></div>

                <div class="divider"></div>

                <div class="section-title">Generated SQL</div>
                <pre id="sql"></pre>

                <div class="section-title" style="margin-top:14px;">Preview</div>
                <div id="preview"></div>

                <div class="muted small" style="margin-top:10px;">
                  Tip: CSV uses the same question; server applies MAX_ROWS from <code>.env</code>.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const qEl = document.getElementById("q");
      const runBtn = document.getElementById("runBtn");
      const statusEl = document.getElementById("status");
      const dotEl = document.getElementById("dot");
      const metaEl = document.getElementById("meta");

      const sqlEl = document.getElementById("sql");
      const previewEl = document.getElementById("preview");
      const errEl = document.getElementById("error");
      const emptyEl = document.getElementById("empty");
      const resultEl = document.getElementById("result");
      const dlBtn = document.getElementById("dlBtn");
      const rowInfoEl = document.getElementById("rowInfo");
      const answerEl = document.getElementById("answer");

      function escapeHtml(s) {
        return (s ?? "").toString()
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#039;");
      }

      function setStatus(text, ok=true){
        statusEl.textContent = text;
        dotEl.style.background = ok ? "#22c55e" : "#ef4444";
        dotEl.style.boxShadow = ok
          ? "0 0 0 3px rgba(34,197,94,.15)"
          : "0 0 0 3px rgba(239,68,68,.15)";
      }

      function renderTable(rows) {
        if (!rows || rows.length === 0) return "<div class='muted small'>No rows returned.</div>";
        const cols = Object.keys(rows[0]);
        let html = "<table><thead><tr>" + cols.map(c => "<th>" + escapeHtml(c) + "</th>").join("") + "</tr></thead><tbody>";
        for (const r of rows) {
          html += "<tr>" + cols.map(c => "<td>" + escapeHtml(r[c]) + "</td>").join("") + "</tr>";
        }
        html += "</tbody></table>";
        return html;
      }

      function clearOutput() {
        dlBtn.style.display = "none";
        dlBtn.onclick = null;

        sqlEl.textContent = "";
        previewEl.innerHTML = "";
        answerEl.textContent = "";

        errEl.style.display = "none";
        errEl.className = "";
        errEl.textContent = "";

        rowInfoEl.textContent = "";

        emptyEl.style.display = "block";
        resultEl.style.display = "none";
      }

      function showError(message) {
        clearOutput();
        errEl.style.display = "block";
        errEl.className = "err";
        errEl.textContent = message || "Unknown error";
        emptyEl.style.display = "none";
      }

      async function run() {
        clearOutput();

        const text = qEl.value.trim();
        if (!text) {
          setStatus("Error", false);
          metaEl.textContent = "Please enter a question.";
          showError("Please enter a question.");
          return;
        }

        setStatus("Running…", true);
        metaEl.textContent = "Sending request…";
        runBtn.disabled = true;

        try {
          const resp = await fetch("/query.debug", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ text })
          });

          const data = await resp.json().catch(() => ({}));
          if (!resp.ok) throw new Error(data.detail || "Request failed");
          emptyEl.style.display = "none";
          resultEl.style.display = "block";

          if (data.type !== "sql") {
            const msg = data.message || "No data available for this query.";
            sqlEl.textContent = data.sql || "";
            answerEl.textContent = msg;
            previewEl.innerHTML = `<div class="muted small">${escapeHtml(msg)}</div>`;
            rowInfoEl.innerHTML = `<span>0 rows in debug response</span>`;
          } else {
            sqlEl.textContent = data.sql || "";
            answerEl.textContent = data.explanation || "";
            previewEl.innerHTML = renderTable(data.rows_preview);
            rowInfoEl.innerHTML = `
              <span>${escapeHtml(String(data.row_count))} rows in debug response</span>
              <span class="muted">·</span>
              <span class="muted">Preview capped at 50</span>
            `;
          }

          dlBtn.style.display = "inline-flex";
          dlBtn.onclick = async () => {
            setStatus("Downloading…", true);
            metaEl.textContent = "Preparing CSV…";

            const r = await fetch("/query.csv", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({ text: qEl.value.trim() })
            });

            if (!r.ok) {
              const j = await r.json().catch(() => ({}));
              throw new Error(j.detail || "CSV request failed");
            }

            const blob = await r.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "query_results.csv";
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 2500);

            setStatus("Done", true);
            metaEl.textContent = "CSV downloaded.";
          };

          setStatus("Done", true);
          metaEl.textContent = "Ready.";
        } catch (e) {
          setStatus("Error", false);
          metaEl.textContent = "Something went wrong.";
          showError(e.message || String(e));
        } finally {
          runBtn.disabled = false;
        }
      }

      runBtn.addEventListener("click", run);
      qEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
          e.preventDefault();
          run();
        }
      });
    </script>
  </body>
</html>
"""
