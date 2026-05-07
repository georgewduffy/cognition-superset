# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=import-outside-toplevel, unused-argument

"""
Targeted regression tests for ``superset/utils/rls.py``.

These tests exercise the authentication-critical RLS predicate path used by
SQL Lab, virtual datasets, and cache key generation. They cover the branches
that the existing ``sql_lab_test.py`` happy-path tests do not, including:

* ``apply_rls`` truthy / falsy return values for the ``has_predicates`` flag.
* ``get_predicates_for_table`` when no dataset matches (returns ``[]``).
* ``get_predicates_for_table`` when the table catalog matches the database's
  default catalog (the ``or_(... is_(None))`` branch).
* ``collect_rls_predicates_for_sql`` happy path including multi-statement and
  multi-table dedup / sort, plus the exception fallback to ``[]``.
"""

from __future__ import annotations

from pytest_mock import MockerFixture

from superset.db_engine_specs.postgres import PostgresEngineSpec
from superset.sql.parse import RLSMethod, SQLStatement, Table
from superset.utils.rls import (
    apply_rls,
    collect_rls_predicates_for_sql,
    get_predicates_for_table,
)


def test_apply_rls_returns_true_when_predicates_exist(
    mocker: MockerFixture,
) -> None:
    """
    ``apply_rls`` returns ``True`` when at least one table has predicates.
    """
    database = mocker.MagicMock()
    database.get_default_catalog.return_value = "examples"
    database.db_engine_spec = PostgresEngineSpec
    mocker.patch(
        "superset.utils.rls.get_predicates_for_table",
        return_value=["c1 = 1"],
    )

    parsed_statement = SQLStatement("SELECT * FROM t1", "postgresql")

    assert apply_rls(database, "examples", "public", parsed_statement) is True


def test_apply_rls_returns_false_when_no_predicates(
    mocker: MockerFixture,
) -> None:
    """
    ``apply_rls`` returns ``False`` when no predicates apply to any table.
    """
    database = mocker.MagicMock()
    database.get_default_catalog.return_value = "examples"
    database.db_engine_spec = PostgresEngineSpec
    mocker.patch(
        "superset.utils.rls.get_predicates_for_table",
        return_value=[],
    )

    parsed_statement = SQLStatement("SELECT * FROM t1", "postgresql")

    assert apply_rls(database, "examples", "public", parsed_statement) is False


def test_apply_rls_uses_engine_spec_method(mocker: MockerFixture) -> None:
    """
    ``apply_rls`` must consult ``db_engine_spec.get_rls_method()`` and forward
    that method to ``parsed_statement.apply_rls``. This is the gate that
    decides whether to inject RLS as a subquery vs. a WHERE-clause predicate.
    Picking the wrong method silently degrades the security posture.
    """
    database = mocker.MagicMock()
    database.get_default_catalog.return_value = "examples"
    database.db_engine_spec.get_rls_method.return_value = RLSMethod.AS_SUBQUERY
    mocker.patch(
        "superset.utils.rls.get_predicates_for_table",
        return_value=["c1 = 1"],
    )

    parsed_statement = mocker.MagicMock()
    parsed_statement.tables = [Table("t1", "public", "examples")]
    parsed_statement.parse_predicate.side_effect = lambda p: p

    apply_rls(database, "examples", "public", parsed_statement)

    database.db_engine_spec.get_rls_method.assert_called_once_with()
    # The method discovered above must be passed through verbatim.
    _, kwargs = parsed_statement.apply_rls.call_args
    args = parsed_statement.apply_rls.call_args.args
    assert RLSMethod.AS_SUBQUERY in args or kwargs.get("method") == (
        RLSMethod.AS_SUBQUERY
    )


def test_get_predicates_for_table_returns_empty_when_no_dataset(
    mocker: MockerFixture,
) -> None:
    """
    ``get_predicates_for_table`` returns ``[]`` when no SqlaTable matches.

    This is critical: missing datasets must not raise, otherwise SQL Lab
    queries against ad-hoc tables would error before security checks run.
    """
    database = mocker.MagicMock()
    db = mocker.patch("superset.utils.rls.db")
    db.session.query().filter().one_or_none.return_value = None

    table = Table("t_unknown", "public", "examples")
    assert get_predicates_for_table(table, database, "examples") == []


def test_get_predicates_for_table_matches_default_catalog_branch(
    mocker: MockerFixture,
) -> None:
    """
    When ``table.catalog == default_catalog`` the catalog predicate must be
    broadened to also match datasets stored with ``catalog IS NULL``.

    Without this branch, datasets created before catalogs were tracked would
    silently lose their RLS rules whenever a query targets the default
    catalog -- a regression risk for OCC-reviewed authentication paths.
    """
    database = mocker.MagicMock()
    dataset = mocker.MagicMock()
    predicate = mocker.MagicMock()
    predicate.compile.return_value = "c1 = 1"
    dataset.get_sqla_row_level_filters.return_value = [predicate]
    db = mocker.patch("superset.utils.rls.db")
    db.session.query().filter().one_or_none.return_value = dataset

    table = Table("t1", "public", "examples")
    # default_catalog matches the table's catalog -> trigger the or_ branch.
    assert get_predicates_for_table(table, database, "examples") == ["c1 = 1"]
    dataset.get_sqla_row_level_filters.assert_called_once_with(
        include_global_guest_rls=False,
    )


def test_collect_rls_predicates_for_sql_happy_path(
    mocker: MockerFixture,
) -> None:
    """
    ``collect_rls_predicates_for_sql`` parses the SQL, qualifies the tables
    with the supplied catalog/schema, and returns the sorted, de-duplicated
    set of predicate strings produced by ``get_predicates_for_table``.

    This is the cache-key path: if predicates collected here drift away from
    those applied at execution time, virtual-dataset cache keys collide
    across users with different RLS rules.
    """
    database = mocker.MagicMock()
    database.db_engine_spec.engine = "postgresql"
    database.get_default_catalog.return_value = "examples"

    def _predicates(table: Table, _database: object, _default: object) -> list[str]:
        return {
            "t1": ["b = 2", "a = 1"],
            "t2": ["a = 1"],  # duplicate across tables
        }.get(table.table, [])

    mocker.patch(
        "superset.utils.rls.get_predicates_for_table",
        side_effect=_predicates,
    )

    result = collect_rls_predicates_for_sql(
        "SELECT * FROM t1 JOIN t2 USING (id)",
        database,
        "examples",
        "public",
    )

    # Must be sorted and de-duplicated -- this contract is what makes the
    # value safe to fold into a cache key.
    assert result == ["a = 1", "b = 2"]


def test_collect_rls_predicates_for_sql_qualifies_tables(
    mocker: MockerFixture,
) -> None:
    """
    Tables in the SQL must be qualified with the caller-supplied catalog and
    schema before predicate lookup; otherwise predicates from the wrong
    tenant could leak into the cache key.
    """
    database = mocker.MagicMock()
    database.db_engine_spec.engine = "postgresql"
    database.get_default_catalog.return_value = "examples"

    get_predicates = mocker.patch(
        "superset.utils.rls.get_predicates_for_table",
        return_value=[],
    )

    collect_rls_predicates_for_sql(
        "SELECT * FROM t1",
        database,
        "examples",
        "public",
    )

    called_tables = [call.args[0] for call in get_predicates.call_args_list]
    assert called_tables == [Table("t1", "public", "examples")]


def test_collect_rls_predicates_for_sql_returns_empty_on_parse_error(
    mocker: MockerFixture,
) -> None:
    """
    If SQL parsing blows up, the function must swallow the exception and
    return ``[]`` so that an unrelated parser regression cannot break
    caching for the entire deployment. This is the documented contract.
    """
    database = mocker.MagicMock()
    database.db_engine_spec.engine = "postgresql"
    database.get_default_catalog.return_value = "examples"

    mocker.patch(
        "superset.sql.parse.SQLScript",
        side_effect=RuntimeError("boom"),
    )

    assert (
        collect_rls_predicates_for_sql(
            "SELECT * FROM t1",
            database,
            "examples",
            "public",
        )
        == []
    )


def test_collect_rls_predicates_for_sql_handles_multiple_statements(
    mocker: MockerFixture,
) -> None:
    """
    Predicates must be collected across every statement in the script and
    de-duplicated, so the cache key reflects the full set of authorisation
    rules that will be applied at execution time.
    """
    database = mocker.MagicMock()
    database.db_engine_spec.engine = "postgresql"
    database.get_default_catalog.return_value = "examples"

    mocker.patch(
        "superset.utils.rls.get_predicates_for_table",
        side_effect=lambda table, *_: {
            "t1": ["x = 1"],
            "t2": ["y = 2"],
            "t3": ["x = 1"],  # duplicate -> deduped
        }.get(table.table, []),
    )

    result = collect_rls_predicates_for_sql(
        "SELECT * FROM t1; SELECT * FROM t2; SELECT * FROM t3",
        database,
        "examples",
        "public",
    )

    assert result == ["x = 1", "y = 2"]
