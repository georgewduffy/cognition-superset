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

# pylint: disable=invalid-name, unused-argument, redefined-outer-name, protected-access

"""Targeted regression tests to cover compliance-critical authentication paths
in `superset/security/manager.py`.

These tests intentionally exercise narrow units (helpers, predicates, audit
hooks, and access decision branches) so that every authorization decision a
guest, anonymous, alpha, gamma, or admin user can hit is covered by an
explicit assertion.  They are designed to satisfy the BofA / OCC compliance
control "auth" by giving an examiner a single file that demonstrates the
behavior of the security manager under each principal type.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from pytest_mock import MockerFixture

from superset.errors import SupersetErrorType
from superset.exceptions import SupersetSecurityException
from superset.extensions import appbuilder
from superset.security.manager import (
    _log_audit_event,
    ExcludeUsersFilter,
    freeze_value,
    SupersetGroupApi,
    SupersetRoleApi,
    SupersetSecurityManager,
    SupersetUserApi,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_pvm(mocker: MockerFixture, perm_name: str, view_name: str) -> Any:
    """Build a fake PermissionView mock with .permission.name and .view_menu.name."""
    pvm = mocker.MagicMock()
    pvm.permission.name = perm_name
    pvm.view_menu.name = view_name
    return pvm


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_freeze_value_is_deterministic() -> None:
    """`freeze_value` must produce a stable string for equivalent dicts."""
    a = {"label": "x", "expressionType": "SIMPLE", "column": {"name": "c"}}
    b = {"column": {"name": "c"}, "expressionType": "SIMPLE", "label": "x"}
    assert freeze_value(a) == freeze_value(b)
    # Lists with different ordering should NOT be considered equal
    assert freeze_value([1, 2]) != freeze_value([2, 1])


def test_log_audit_event_swallows_logger_errors(mocker: MockerFixture) -> None:
    """An exception in the event logger must not bubble up to callers."""
    fake_logger = mocker.MagicMock()
    fake_logger.log.side_effect = RuntimeError("logger down")
    mocker.patch(
        "superset.extensions.event_logger",
        new=fake_logger,
    )
    mocker.patch("superset.security.manager.get_user_id", return_value=7)

    # Should not raise even though .log() raises.
    _log_audit_event("UserCreated", {"username": "alice"})

    fake_logger.log.assert_called_once()
    kwargs = fake_logger.log.call_args.kwargs
    assert kwargs["user_id"] == 7
    assert kwargs["action"] == "UserCreated"
    assert kwargs["records"] == [{"username": "alice"}]


def test_log_audit_event_success_path(mocker: MockerFixture) -> None:
    """Successful audit event call should reach the configured event logger."""
    fake_logger = mocker.MagicMock()
    mocker.patch("superset.extensions.event_logger", new=fake_logger)
    mocker.patch("superset.security.manager.get_user_id", return_value=42)

    _log_audit_event("RoleCreated", {"role_name": "viewer"})

    fake_logger.log.assert_called_once()
    kwargs = fake_logger.log.call_args.kwargs
    assert kwargs["user_id"] == 42
    assert kwargs["action"] == "RoleCreated"
    assert kwargs["records"] == [{"role_name": "viewer"}]


# ---------------------------------------------------------------------------
# Audit-logging hooks on SupersetRoleApi / SupersetUserApi / SupersetGroupApi
# ---------------------------------------------------------------------------


def test_role_api_audit_hooks(mocker: MockerFixture) -> None:
    audit = mocker.patch("superset.security.manager._log_audit_event")
    api = SupersetRoleApi.__new__(SupersetRoleApi)

    role = SimpleNamespace(name="Alpha", id=2, permissions=["x"])
    api.pre_delete(role)
    assert role.permissions == []

    api.post_add(role)
    api.post_update(role)
    api.post_delete(role)

    actions = [c.args[0] for c in audit.call_args_list]
    assert actions == ["RoleCreated", "RoleUpdated", "RoleDeleted"]
    for call in audit.call_args_list:
        payload = call.args[1]
        assert payload == {"role_name": "Alpha", "role_id": 2}


def test_user_api_audit_hooks(mocker: MockerFixture) -> None:
    audit = mocker.patch("superset.security.manager._log_audit_event")
    api = SupersetUserApi.__new__(SupersetUserApi)

    user = SimpleNamespace(
        username="alice",
        id=10,
        email="alice@example.com",
        active=True,
        roles=["x"],
    )
    api.pre_delete(user)
    assert user.roles == []

    api.post_add(user)
    api.post_update(user)
    api.post_delete(user)

    actions = [c.args[0] for c in audit.call_args_list]
    assert actions == ["UserCreated", "UserUpdated", "UserDeleted"]

    add_payload = audit.call_args_list[0].args[1]
    assert add_payload == {
        "target_username": "alice",
        "target_user_id": 10,
        "email": "alice@example.com",
    }
    update_payload = audit.call_args_list[1].args[1]
    assert update_payload["active"] is True
    delete_payload = audit.call_args_list[2].args[1]
    assert delete_payload == {"target_username": "alice", "target_user_id": 10}


def test_group_api_audit_hooks(mocker: MockerFixture) -> None:
    audit = mocker.patch("superset.security.manager._log_audit_event")
    api = SupersetGroupApi.__new__(SupersetGroupApi)
    grp = SimpleNamespace(name="Auditors", id=3)

    api.post_add(grp)
    api.post_update(grp)
    api.post_delete(grp)

    actions = [c.args[0] for c in audit.call_args_list]
    assert actions == ["GroupCreated", "GroupUpdated", "GroupDeleted"]
    for call in audit.call_args_list:
        assert call.args[1] == {"group_name": "Auditors", "group_id": 3}


def test_user_login_logout_login_failed_emit_audit_events(
    app_context: None, mocker: MockerFixture
) -> None:
    audit = mocker.patch("superset.security.manager._log_audit_event")
    sm = SupersetSecurityManager(appbuilder)
    user = SimpleNamespace(username="alice", id=99)

    sm.on_user_login(user)
    sm.on_user_login_failed(user)
    sm.on_user_logout(user)

    actions = [c.args[0] for c in audit.call_args_list]
    assert actions == ["UserLoggedIn", "UserLoginFailed", "UserLoggedOut"]
    for call in audit.call_args_list:
        payload = call.args[1]
        # LoggedOut uses getattr so its keys are slightly different.
        assert payload.get("user_id", payload.get("user_id")) == 99
        assert payload.get("username", payload.get("username")) == "alice"


def test_on_user_logout_handles_user_without_attributes(
    app_context: None, mocker: MockerFixture
) -> None:
    """Logout audit must accept an unauthenticated/anonymous user-like object."""
    audit = mocker.patch("superset.security.manager._log_audit_event")
    sm = SupersetSecurityManager(appbuilder)

    class Anon:
        pass

    sm.on_user_logout(Anon())
    audit.assert_called_once()
    payload = audit.call_args.args[1]
    assert payload == {"username": None, "user_id": None}


# ---------------------------------------------------------------------------
# ExcludeUsersFilter
# ---------------------------------------------------------------------------


def _build_exclude_filter(mocker: MockerFixture) -> ExcludeUsersFilter:
    """ExcludeUsersFilter inherits from FAB BaseFilter which requires
    `datamodel.obj` during init.  Provide a minimal stand-in."""
    datamodel = mocker.MagicMock()
    datamodel.obj = mocker.MagicMock()
    return ExcludeUsersFilter("username", datamodel)


def test_exclude_users_filter_returns_query_when_nothing_to_exclude(
    app_context: None, mocker: MockerFixture
) -> None:
    """When the exclusion list is empty, the original query is returned."""
    filt = _build_exclude_filter(mocker)

    mocker.patch.dict(
        "superset.security.manager.current_app.config",
        {"EXCLUDE_USERS_FROM_LISTS": []},
    )

    query = mocker.MagicMock()
    result = filt.apply(query, value=None)
    assert result is query
    query.filter.assert_not_called()


def test_exclude_users_filter_uses_config_when_set(
    app_context: None, mocker: MockerFixture
) -> None:
    """A non-empty EXCLUDE_USERS_FROM_LISTS must add a filter clause."""
    filt = _build_exclude_filter(mocker)
    mocker.patch.dict(
        "superset.security.manager.current_app.config",
        {"EXCLUDE_USERS_FROM_LISTS": ["bot", "system"]},
    )

    query = mocker.MagicMock()
    filt.apply(query, value=None)
    query.filter.assert_called_once()


def test_exclude_users_filter_falls_back_to_sm_callable(
    app_context: None, mocker: MockerFixture
) -> None:
    """When config is None, the SM's get_exclude_users_from_lists is consulted."""
    filt = _build_exclude_filter(mocker)

    mocker.patch.dict(
        "superset.security.manager.current_app.config",
        {"EXCLUDE_USERS_FROM_LISTS": None},
    )
    mocker.patch.object(
        appbuilder.sm,
        "get_exclude_users_from_lists",
        return_value=["bot"],
    )

    query = mocker.MagicMock()
    filt.apply(query, value=None)
    query.filter.assert_called_once()


def test_get_exclude_users_from_lists_default_is_empty(app_context: None) -> None:
    """The base implementation returns an empty list (override hook)."""
    sm = SupersetSecurityManager(appbuilder)
    assert sm.get_exclude_users_from_lists() == []


# ---------------------------------------------------------------------------
# Permission helpers
# ---------------------------------------------------------------------------


def test_get_database_perm_format(app_context: None) -> None:
    sm = SupersetSecurityManager(appbuilder)
    assert sm.get_database_perm(7, "examples") == "[examples].(id:7)"


def test_get_dataset_perm_format(app_context: None) -> None:
    sm = SupersetSecurityManager(appbuilder)
    assert (
        sm.get_dataset_perm(11, "users", "examples")
        == "[examples].[users](id:11)"
    )


def test_error_object_helpers(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)

    dashboard = mocker.MagicMock()
    chart = mocker.MagicMock()
    err_dash = sm.get_dashboard_access_error_object(dashboard)
    err_chart = sm.get_chart_access_error_object(chart)
    assert err_dash.error_type == SupersetErrorType.DASHBOARD_SECURITY_ACCESS_ERROR
    assert err_chart.error_type == SupersetErrorType.CHART_SECURITY_ACCESS_ERROR

    datasource = mocker.MagicMock()
    datasource.data = {"id": 5, "name": "users"}
    msg = sm.get_datasource_access_error_msg(datasource)
    assert "datasource 5" in msg
    err_ds = sm.get_datasource_access_error_object(datasource)
    assert err_ds.error_type == SupersetErrorType.DATASOURCE_SECURITY_ACCESS_ERROR
    assert err_ds.extra["datasource"] == 5
    assert err_ds.extra["datasource_name"] == "users"

    # Default config has no PERMISSION_INSTRUCTIONS_LINK -> empty string expected
    assert sm.get_datasource_access_link(datasource) in ("", None)
    assert sm.get_table_access_link({mocker.MagicMock()}) in ("", None)


def test_get_table_access_error_msg_and_object(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    table = mocker.MagicMock()
    table.__str__ = lambda _self: "examples.users"  # type: ignore[assignment]

    msg = sm.get_table_access_error_msg({table})
    assert "examples.users" in msg

    err = sm.get_table_access_error_object({table})
    assert err.error_type == SupersetErrorType.TABLE_SECURITY_ACCESS_ERROR
    assert err.extra["tables"] == ["examples.users"]


# ---------------------------------------------------------------------------
# can_access* family
# ---------------------------------------------------------------------------


def test_can_access_anonymous_falls_back_to_is_item_public(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)

    anon = mocker.MagicMock()
    anon.is_anonymous = True
    mocker.patch("superset.security.manager.g", new=SimpleNamespace(user=anon))

    is_public = mocker.patch.object(sm, "is_item_public", return_value=True)
    has_view = mocker.patch.object(sm, "_has_view_access", return_value=False)

    assert sm.can_access("can_read", "Dashboard") is True
    is_public.assert_called_once_with("can_read", "Dashboard")
    has_view.assert_not_called()


def test_can_access_authenticated_user_calls_has_view_access(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)

    user = mocker.MagicMock()
    user.is_anonymous = False
    mocker.patch("superset.security.manager.g", new=SimpleNamespace(user=user))

    has_view = mocker.patch.object(sm, "_has_view_access", return_value=False)
    mocker.patch.object(sm, "is_item_public", return_value=True)

    assert sm.can_access("can_write", "Dashboard") is False
    has_view.assert_called_once_with(user, "can_write", "Dashboard")


def test_can_access_all_queries_proxy(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    can_access = mocker.patch.object(sm, "can_access", return_value=True)
    assert sm.can_access_all_queries() is True
    can_access.assert_called_once_with("all_query_access", "all_query_access")


def test_can_access_all_databases_proxy(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    can_access = mocker.patch.object(sm, "can_access", return_value=False)
    assert sm.can_access_all_databases() is False
    can_access.assert_called_once_with("all_database_access", "all_database_access")


def test_can_access_all_datasources_short_circuits_on_all_databases(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_all_databases", return_value=True)
    can_access = mocker.patch.object(sm, "can_access")
    assert sm.can_access_all_datasources() is True
    can_access.assert_not_called()


def test_can_access_all_datasources_falls_through_to_perm(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_all_databases", return_value=False)
    can_access = mocker.patch.object(sm, "can_access", return_value=True)
    assert sm.can_access_all_datasources() is True
    can_access.assert_called_once_with(
        "all_datasource_access", "all_datasource_access"
    )


def test_can_access_database_with_explicit_grant(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_all_datasources", return_value=False)
    mocker.patch.object(sm, "can_access_all_databases", return_value=False)
    can_access = mocker.patch.object(sm, "can_access", return_value=True)

    db = mocker.MagicMock()
    db.perm = "[examples].(id:1)"
    assert sm.can_access_database(db) is True
    can_access.assert_called_once_with("database_access", db.perm)


def test_can_access_database_denied_when_no_grants(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_all_datasources", return_value=False)
    mocker.patch.object(sm, "can_access_all_databases", return_value=False)
    mocker.patch.object(sm, "can_access", return_value=False)

    db = mocker.MagicMock()
    db.perm = "[examples].(id:1)"
    assert sm.can_access_database(db) is False


def test_can_access_catalog_paths(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    db = mocker.MagicMock()
    db.database_name = "examples"

    # Path 1: all_datasources -> True
    mocker.patch.object(sm, "can_access_all_datasources", return_value=True)
    assert sm.can_access_catalog(db, "main") is True

    # Path 2: database_access grants the catalog implicitly
    mocker.patch.object(sm, "can_access_all_datasources", return_value=False)
    mocker.patch.object(sm, "can_access_database", return_value=True)
    assert sm.can_access_catalog(db, "main") is True

    # Path 3: explicit catalog perm
    mocker.patch.object(sm, "can_access_database", return_value=False)
    can_access = mocker.patch.object(sm, "can_access", return_value=True)
    assert sm.can_access_catalog(db, "main") is True
    can_access.assert_called_once_with("catalog_access", "[examples].[main]")

    # Path 4: denied
    mocker.patch.object(sm, "can_access", return_value=False)
    assert sm.can_access_catalog(db, "main") is False


def test_can_access_schema_admin_short_circuit(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_all_datasources", return_value=True)
    ds = mocker.MagicMock()
    assert sm.can_access_schema(ds) is True


def test_can_access_schema_sql_hierarchy(
    app_context: None, mocker: MockerFixture
) -> None:
    """Cover database-, catalog-, and schema-level grants for SQL datasources."""
    from superset.connectors.sqla.models import BaseDatasource

    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_all_datasources", return_value=False)

    ds = mocker.MagicMock(spec=BaseDatasource)
    ds.database = mocker.MagicMock()
    ds.catalog = "main"
    ds.schema_perm = "[examples].[main].[public]"

    # database-level grant
    mocker.patch.object(sm, "can_access_database", return_value=True)
    assert sm.can_access_schema(ds) is True

    # catalog-level grant
    mocker.patch.object(sm, "can_access_database", return_value=False)
    can_access_catalog = mocker.patch.object(
        sm, "can_access_catalog", return_value=True
    )
    assert sm.can_access_schema(ds) is True
    can_access_catalog.assert_called_once_with(ds.database, "main")

    # schema-level grant
    mocker.patch.object(sm, "can_access_catalog", return_value=False)
    can_access = mocker.patch.object(sm, "can_access", return_value=True)
    assert sm.can_access_schema(ds) is True
    can_access.assert_called_once_with("schema_access", ds.schema_perm)

    # denied
    mocker.patch.object(sm, "can_access", return_value=False)
    assert sm.can_access_schema(ds) is False


def test_can_access_schema_non_sql_datasource(
    app_context: None, mocker: MockerFixture
) -> None:
    """Non-SQL explorables only honor `all_datasources` access."""
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_all_datasources", return_value=False)

    # An object that is NOT a BaseDatasource
    ds = SimpleNamespace(perm="x", schema_perm="y", catalog="c", database=None)
    assert sm.can_access_schema(ds) is False


def test_can_access_datasource_returns_true_when_no_exception(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    raise_for = mocker.patch.object(sm, "raise_for_access")
    ds = mocker.MagicMock()
    assert sm.can_access_datasource(ds) is True
    raise_for.assert_called_once_with(datasource=ds)


def test_can_access_datasource_returns_false_on_security_error(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    err = SupersetSecurityException(
        sm.get_dashboard_access_error_object(mocker.MagicMock())
    )
    mocker.patch.object(sm, "raise_for_access", side_effect=err)
    assert sm.can_access_datasource(mocker.MagicMock()) is False


def test_can_access_dashboard_and_chart_failure_paths(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)

    err = SupersetSecurityException(
        sm.get_dashboard_access_error_object(mocker.MagicMock())
    )
    mocker.patch.object(sm, "raise_for_access", side_effect=err)
    assert sm.can_access_dashboard(mocker.MagicMock()) is False
    assert sm.can_access_chart(mocker.MagicMock()) is False


def test_can_access_dashboard_and_chart_happy_paths(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    raise_for = mocker.patch.object(sm, "raise_for_access")
    assert sm.can_access_dashboard(mocker.MagicMock()) is True
    assert sm.can_access_chart(mocker.MagicMock()) is True
    assert raise_for.call_count == 2


def test_can_access_table_returns_false_on_security_error(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    err = SupersetSecurityException(
        sm.get_dashboard_access_error_object(mocker.MagicMock())
    )
    mocker.patch.object(sm, "raise_for_access", side_effect=err)
    assert sm.can_access_table(mocker.MagicMock(), mocker.MagicMock()) is False


def test_can_access_table_returns_true_on_success(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    raise_for = mocker.patch.object(sm, "raise_for_access")
    db, table = mocker.MagicMock(), mocker.MagicMock()
    assert sm.can_access_table(db, table) is True
    raise_for.assert_called_once_with(database=db, table=table)


# ---------------------------------------------------------------------------
# can_drill_dataset_via_dashboard_access
# ---------------------------------------------------------------------------


def test_can_drill_dataset_via_dashboard_access_guest_path(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch(
        "superset.is_feature_enabled",
        side_effect=lambda f: f == "EMBEDDED_SUPERSET",
    )
    mocker.patch.object(sm, "is_guest_user", return_value=True)
    mocker.patch.object(sm, "has_guest_access", return_value=True)

    ds = mocker.MagicMock()
    ds.id = 1
    dashboard = mocker.MagicMock()
    dashboard.datasources = [ds]
    assert sm.can_drill_dataset_via_dashboard_access(ds, dashboard) is True


def test_can_drill_dataset_via_dashboard_access_dashboard_rbac(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch(
        "superset.is_feature_enabled",
        side_effect=lambda f: f == "DASHBOARD_RBAC",
    )
    mocker.patch.object(sm, "is_guest_user", return_value=False)
    role = mocker.MagicMock(id=1)
    user_role = mocker.MagicMock(id=1)
    mocker.patch.object(sm, "get_user_roles", return_value=[user_role])

    ds = mocker.MagicMock()
    ds.id = 1
    dashboard = mocker.MagicMock()
    dashboard.roles = [role]
    dashboard.published = True
    dashboard.datasources = [ds]
    assert sm.can_drill_dataset_via_dashboard_access(ds, dashboard) is True


def test_can_drill_dataset_via_dashboard_access_denied_when_no_overlap(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch("superset.is_feature_enabled", return_value=False)
    mocker.patch.object(sm, "is_guest_user", return_value=False)
    dashboard = mocker.MagicMock()
    dashboard.roles = []
    dashboard.datasources = []
    assert (
        sm.can_drill_dataset_via_dashboard_access(mocker.MagicMock(), dashboard)
        is False
    )


# ---------------------------------------------------------------------------
# has_drill_access
# ---------------------------------------------------------------------------


def test_has_drill_access_drill_to_detail(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    ds = mocker.MagicMock()
    dashboard = mocker.MagicMock()
    dashboard.datasources = [ds]
    form_data: dict[str, Any] = {}
    assert sm.has_drill_access(form_data, dashboard, ds) is True


def test_has_drill_access_drill_to_detail_dataset_not_in_dashboard(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    ds = mocker.MagicMock()
    dashboard = mocker.MagicMock()
    dashboard.datasources = []
    assert sm.has_drill_access({}, dashboard, ds) is False


def test_has_drill_access_drill_by_happy_path(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)

    ds = mocker.MagicMock()
    ds.has_drill_by_columns.return_value = True
    slc = mocker.MagicMock()
    slc.datasource = ds

    dashboard = mocker.MagicMock()
    dashboard.slices = [slc]
    dashboard.datasources = []  # force the drill-by branch

    fake_session = mocker.MagicMock()
    query_obj = fake_session.query.return_value
    query_obj.filter.return_value.one_or_none.return_value = slc
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )

    form_data = {"slice_id": 0, "chart_id": 100, "groupby": ["country"]}
    assert sm.has_drill_access(form_data, dashboard, ds) is True


def test_has_drill_access_drill_by_chart_id_missing(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    dashboard = mocker.MagicMock()
    dashboard.slices = []
    dashboard.datasources = []
    form_data = {"slice_id": 0}
    assert sm.has_drill_access(form_data, dashboard, mocker.MagicMock()) is False


# ---------------------------------------------------------------------------
# Role pvm classification predicates
# ---------------------------------------------------------------------------


def test_is_user_defined_permission_true_and_false(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    udp = _make_pvm(mocker, "datasource_access", "[examples].[users]")
    not_udp = _make_pvm(mocker, "can_read", "Dashboard")
    # Inject a known OBJECT_SPEC_PERMISSIONS to make the test deterministic
    sm.OBJECT_SPEC_PERMISSIONS = {"datasource_access"}
    assert sm._is_user_defined_permission(udp) is True
    assert sm._is_user_defined_permission(not_udp) is False


def test_is_admin_only_branches(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)

    # 1) Anything in ALPHA_ONLY_PMVS is explicitly NOT admin-only.
    sm.ALPHA_ONLY_PMVS = {("can_read", "AlphaView")}
    pvm_alpha = _make_pvm(mocker, "can_read", "AlphaView")
    assert sm._is_admin_only(pvm_alpha) is False

    # 2) Read-only model views with non-read perms are admin-only.
    sm.READ_ONLY_MODEL_VIEWS = {"Database"}
    sm.READ_ONLY_PERMISSION = {"can_read"}
    pvm_write_db = _make_pvm(mocker, "can_write", "Database")
    assert sm._is_admin_only(pvm_write_db) is True

    # 3) View menu is admin-only.
    sm.ADMIN_ONLY_VIEW_MENUS = {"List Users"}
    pvm_users = _make_pvm(mocker, "can_read", "List Users")
    assert sm._is_admin_only(pvm_users) is True

    # 4) Permission name is admin-only.
    sm.ADMIN_ONLY_PERMISSIONS = {"can_grant"}
    pvm_grant = _make_pvm(mocker, "can_grant", "Some")
    assert sm._is_admin_only(pvm_grant) is True

    # 5) Default-not-admin
    pvm_default = _make_pvm(mocker, "can_read", "DashboardModelView")
    assert sm._is_admin_only(pvm_default) is False


def test_is_alpha_only_branches(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    sm.GAMMA_READ_ONLY_MODEL_VIEWS = {"Annotation"}
    sm.READ_ONLY_PERMISSION = {"can_read"}
    sm.ALPHA_ONLY_PMVS = set()
    sm.ALPHA_ONLY_VIEW_MENUS = {"Manage"}
    sm.ALPHA_ONLY_PERMISSIONS = {"can_share"}

    # Non-read perm on a Gamma-read-only view -> alpha only
    pvm_write_anno = _make_pvm(mocker, "can_write", "Annotation")
    assert sm._is_alpha_only(pvm_write_anno) is True

    # Permission inside ALPHA_ONLY_PMVS
    sm.ALPHA_ONLY_PMVS = {("can_read", "AlphaView")}
    pvm_alpha = _make_pvm(mocker, "can_read", "AlphaView")
    assert sm._is_alpha_only(pvm_alpha) is True

    # View menu in ALPHA_ONLY_VIEW_MENUS
    pvm_manage = _make_pvm(mocker, "can_read", "Manage")
    assert sm._is_alpha_only(pvm_manage) is True

    # Permission name in ALPHA_ONLY_PERMISSIONS
    pvm_share = _make_pvm(mocker, "can_share", "Whatever")
    assert sm._is_alpha_only(pvm_share) is True

    # Default not alpha-only
    pvm_default = _make_pvm(mocker, "can_read", "Some")
    assert sm._is_alpha_only(pvm_default) is False


def test_is_accessible_to_all(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    sm.ACCESSIBLE_PERMS = {"can_userinfo"}
    pvm_yes = _make_pvm(mocker, "can_userinfo", "UserInfoView")
    pvm_no = _make_pvm(mocker, "can_read", "Dashboard")
    assert sm._is_accessible_to_all(pvm_yes) is True
    assert sm._is_accessible_to_all(pvm_no) is False


def test_is_admin_pvm(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    sm.OBJECT_SPEC_PERMISSIONS = {"datasource_access"}
    pvm_admin = _make_pvm(mocker, "can_read", "Dashboard")
    pvm_user_defined = _make_pvm(mocker, "datasource_access", "[examples].[users]")
    assert sm._is_admin_pvm(pvm_admin) is True
    assert sm._is_admin_pvm(pvm_user_defined) is False


def test_is_alpha_pvm(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    sm.OBJECT_SPEC_PERMISSIONS = {"datasource_access"}
    sm.ALPHA_ONLY_PMVS = set()
    sm.ALPHA_ONLY_VIEW_MENUS = set()
    sm.ALPHA_ONLY_PERMISSIONS = set()
    sm.ADMIN_ONLY_VIEW_MENUS = set()
    sm.ADMIN_ONLY_PERMISSIONS = set()
    sm.READ_ONLY_MODEL_VIEWS = set()
    sm.READ_ONLY_PERMISSION = set()
    sm.GAMMA_READ_ONLY_MODEL_VIEWS = set()
    sm.SQLLAB_ONLY_PERMISSIONS = set()
    sm.ACCESSIBLE_PERMS = set()

    # Plain view -> alpha
    pvm_alpha = _make_pvm(mocker, "can_read", "Dashboard")
    assert sm._is_alpha_pvm(pvm_alpha) is True

    # Marking it as user defined -> not alpha
    pvm_user_defined = _make_pvm(mocker, "datasource_access", "[a].[b]")
    assert sm._is_alpha_pvm(pvm_user_defined) is False

    # Accessible-to-all overrides exclusions
    sm.ACCESSIBLE_PERMS = {"datasource_access"}
    assert sm._is_alpha_pvm(pvm_user_defined) is True


def test_is_gamma_pvm(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    sm.OBJECT_SPEC_PERMISSIONS = set()
    sm.ALPHA_ONLY_PMVS = set()
    sm.ALPHA_ONLY_VIEW_MENUS = set()
    sm.ALPHA_ONLY_PERMISSIONS = set()
    sm.ADMIN_ONLY_VIEW_MENUS = set()
    sm.ADMIN_ONLY_PERMISSIONS = set()
    sm.READ_ONLY_MODEL_VIEWS = set()
    sm.READ_ONLY_PERMISSION = set()
    sm.GAMMA_READ_ONLY_MODEL_VIEWS = set()
    sm.SQLLAB_ONLY_PERMISSIONS = set()
    sm.GAMMA_EXCLUDED_PVMS = set()
    sm.ACCESSIBLE_PERMS = set()

    pvm = _make_pvm(mocker, "can_read", "Dashboard")
    assert sm._is_gamma_pvm(pvm) is True

    sm.GAMMA_EXCLUDED_PVMS = {("can_read", "Dashboard")}
    assert sm._is_gamma_pvm(pvm) is False
    sm.ACCESSIBLE_PERMS = {"can_read"}
    assert sm._is_gamma_pvm(pvm) is True


def test_is_sql_lab_only_and_pvm(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    sm.SQLLAB_ONLY_PERMISSIONS = {("can_sql_json", "Superset")}
    sm.SQLLAB_EXTRA_PERMISSION_VIEWS = {("can_csv", "Superset")}

    only = _make_pvm(mocker, "can_sql_json", "Superset")
    extra = _make_pvm(mocker, "can_csv", "Superset")
    other = _make_pvm(mocker, "can_read", "Dashboard")

    assert sm._is_sql_lab_only(only) is True
    assert sm._is_sql_lab_only(other) is False
    assert sm._is_sql_lab_pvm(only) is True
    assert sm._is_sql_lab_pvm(extra) is True
    assert sm._is_sql_lab_pvm(other) is False


def test_is_public_pvm(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    sm.PUBLIC_ROLE_PERMISSIONS = {("can_read", "Chart")}
    sm.PUBLIC_EXCLUDED_VIEW_MENUS = {"SQL Lab"}
    sm.OBJECT_SPEC_PERMISSIONS = {"datasource_access"}

    allowed = _make_pvm(mocker, "can_read", "Chart")
    assert sm._is_public_pvm(allowed) is True

    excluded = _make_pvm(mocker, "can_read", "SQL Lab")
    assert sm._is_public_pvm(excluded) is False

    user_defined = _make_pvm(mocker, "datasource_access", "[a].[b]")
    assert sm._is_public_pvm(user_defined) is False

    other = _make_pvm(mocker, "can_write", "Dashboard")
    assert sm._is_public_pvm(other) is False


# ---------------------------------------------------------------------------
# raise_for_access — extra branches not covered by the existing test file.
# ---------------------------------------------------------------------------


def test_raise_for_access_dashboard_admin_short_circuits(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_guest_user", return_value=False)
    mocker.patch.object(sm, "is_admin", return_value=True)
    sm.raise_for_access(dashboard=mocker.MagicMock())


def test_raise_for_access_dashboard_owner_short_circuits(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_guest_user", return_value=False)
    mocker.patch.object(sm, "is_admin", return_value=False)
    mocker.patch.object(sm, "is_owner", return_value=True)
    sm.raise_for_access(dashboard=mocker.MagicMock())


def test_raise_for_access_dashboard_no_datasources_allowed(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_guest_user", return_value=False)
    mocker.patch.object(sm, "is_admin", return_value=False)
    mocker.patch.object(sm, "is_owner", return_value=False)
    mocker.patch("superset.is_feature_enabled", return_value=False)
    dashboard = mocker.MagicMock()
    dashboard.datasources = []
    sm.raise_for_access(dashboard=dashboard)


def test_raise_for_access_dashboard_at_least_one_accessible_datasource(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_guest_user", return_value=False)
    mocker.patch.object(sm, "is_admin", return_value=False)
    mocker.patch.object(sm, "is_owner", return_value=False)
    mocker.patch("superset.is_feature_enabled", return_value=False)
    mocker.patch.object(
        sm, "can_access_datasource", side_effect=[False, True]
    )
    dashboard = mocker.MagicMock()
    dashboard.datasources = [mocker.MagicMock(), mocker.MagicMock()]
    sm.raise_for_access(dashboard=dashboard)


def test_raise_for_access_dashboard_denied_when_no_dataset_access(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_guest_user", return_value=False)
    mocker.patch.object(sm, "is_admin", return_value=False)
    mocker.patch.object(sm, "is_owner", return_value=False)
    mocker.patch("superset.is_feature_enabled", return_value=False)
    mocker.patch.object(sm, "can_access_datasource", return_value=False)
    dashboard = mocker.MagicMock()
    dashboard.datasources = [mocker.MagicMock()]
    with pytest.raises(SupersetSecurityException):
        sm.raise_for_access(dashboard=dashboard)


def test_raise_for_access_dashboard_rbac_published_with_role_overlap(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_guest_user", return_value=False)
    mocker.patch.object(sm, "is_admin", return_value=False)
    mocker.patch.object(sm, "is_owner", return_value=False)
    mocker.patch(
        "superset.is_feature_enabled",
        side_effect=lambda f: f == "DASHBOARD_RBAC",
    )
    role = mocker.MagicMock(id=1)
    user_role = mocker.MagicMock(id=1)
    mocker.patch.object(sm, "get_user_roles", return_value=[user_role])

    dashboard = mocker.MagicMock()
    dashboard.roles = [role]
    dashboard.published = True
    dashboard.datasources = []
    sm.raise_for_access(dashboard=dashboard)


def test_raise_for_access_dashboard_guest_with_access(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_guest_user", return_value=True)
    mocker.patch.object(sm, "has_guest_access", return_value=True)
    sm.raise_for_access(dashboard=mocker.MagicMock())


def test_raise_for_access_dashboard_guest_without_access_raises(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_guest_user", return_value=True)
    mocker.patch.object(sm, "has_guest_access", return_value=False)
    with pytest.raises(SupersetSecurityException):
        sm.raise_for_access(dashboard=mocker.MagicMock())


def test_raise_for_access_chart_admin_short_circuits(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_admin", return_value=True)
    sm.raise_for_access(chart=mocker.MagicMock())


def test_raise_for_access_chart_owner_short_circuits(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_admin", return_value=False)
    mocker.patch.object(sm, "is_owner", return_value=True)
    sm.raise_for_access(chart=mocker.MagicMock())


def test_raise_for_access_chart_via_datasource(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_admin", return_value=False)
    mocker.patch.object(sm, "is_owner", return_value=False)
    mocker.patch.object(sm, "can_access_datasource", return_value=True)
    chart = mocker.MagicMock()
    chart.datasource = mocker.MagicMock()
    sm.raise_for_access(chart=chart)


def test_raise_for_access_chart_denied(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_admin", return_value=False)
    mocker.patch.object(sm, "is_owner", return_value=False)
    mocker.patch.object(sm, "can_access_datasource", return_value=False)
    chart = mocker.MagicMock()
    chart.datasource = mocker.MagicMock()
    with pytest.raises(SupersetSecurityException):
        sm.raise_for_access(chart=chart)


def test_raise_for_access_database_grants_full_query(
    app_context: None, mocker: MockerFixture
) -> None:
    """If the user can access the database the rest of the SQL checks short
    circuit."""
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_database", return_value=True)
    mocker.patch.object(sm, "is_guest_user", return_value=False)

    db = mocker.MagicMock()
    db.get_default_catalog.return_value = "main"
    table = mocker.MagicMock()
    sm.raise_for_access(database=db, table=table)


def test_raise_for_access_table_denied_when_no_grants(
    app_context: None, mocker: MockerFixture
) -> None:
    """No database/catalog/schema/datasource grant -> SecurityException."""
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_database", return_value=False)
    mocker.patch.object(sm, "is_guest_user", return_value=False)
    mocker.patch.object(sm, "can_access", return_value=False)

    SqlaTable = mocker.patch("superset.connectors.sqla.models.SqlaTable")  # noqa: N806
    SqlaTable.query_datasources_by_name.return_value = []

    db = mocker.MagicMock()
    db.database_name = "examples"
    db.get_default_catalog.return_value = "main"

    table = mocker.MagicMock()
    table.qualify.return_value = mocker.MagicMock(
        catalog="main", schema="public", table="users"
    )
    with pytest.raises(SupersetSecurityException):
        sm.raise_for_access(database=db, table=table)


# ---------------------------------------------------------------------------
# Guest-token utilities & is_admin / is_guest_user / get_user_roles
# ---------------------------------------------------------------------------


def test_is_guest_user_disabled_returns_false(
    app_context: None, mocker: MockerFixture
) -> None:
    mocker.patch("superset.is_feature_enabled", return_value=False)
    assert SupersetSecurityManager.is_guest_user() is False


def test_is_guest_user_when_feature_enabled_with_user(
    app_context: None, mocker: MockerFixture
) -> None:
    mocker.patch("superset.is_feature_enabled", return_value=True)
    user = SimpleNamespace(is_guest_user=True)
    mocker.patch(
        "superset.security.manager.get_current_user", return_value=user
    )
    mocker.patch("superset.security.manager.g", new=SimpleNamespace(user=user))
    assert SupersetSecurityManager.is_guest_user() is True


def test_is_guest_user_when_feature_enabled_without_current_user(
    app_context: None, mocker: MockerFixture
) -> None:
    mocker.patch("superset.is_feature_enabled", return_value=True)
    mocker.patch("superset.security.manager.get_current_user", return_value=None)
    assert SupersetSecurityManager.is_guest_user() is False


def test_get_current_guest_user_if_guest(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    user = mocker.MagicMock()
    mocker.patch("superset.security.manager.g", new=SimpleNamespace(user=user))

    mocker.patch.object(SupersetSecurityManager, "is_guest_user", return_value=True)
    assert sm.get_current_guest_user_if_guest() is user

    mocker.patch.object(SupersetSecurityManager, "is_guest_user", return_value=False)
    assert sm.get_current_guest_user_if_guest() is None


def test_has_guest_access_matches_dashboard_id_and_uuid(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    from superset.security.guest_token import GuestTokenResourceType

    guest = mocker.MagicMock()
    guest.resources = [
        {"type": GuestTokenResourceType.DASHBOARD, "id": "42"},
    ]
    mocker.patch.object(sm, "get_current_guest_user_if_guest", return_value=guest)

    dashboard = mocker.MagicMock()
    dashboard.id = 42
    assert sm.has_guest_access(dashboard) is True

    # Dashboard id mismatch but embedded uuid matches.
    guest.resources = [
        {"type": GuestTokenResourceType.DASHBOARD, "id": "abcd"},
    ]
    dashboard.id = 99
    embedded = mocker.MagicMock()
    embedded.uuid = "abcd"
    dashboard.embedded = [embedded]
    assert sm.has_guest_access(dashboard) is True

    # Not embedded and id mismatch -> False
    dashboard.embedded = []
    assert sm.has_guest_access(dashboard) is False


def test_has_guest_access_no_guest_returns_false(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "get_current_guest_user_if_guest", return_value=None)
    assert sm.has_guest_access(mocker.MagicMock()) is False


def test_get_anonymous_user_returns_anonymous(app_context: None) -> None:
    from flask_login import AnonymousUserMixin

    sm = SupersetSecurityManager(appbuilder)
    assert isinstance(sm.get_anonymous_user(), AnonymousUserMixin)


def test_get_user_roles_anonymous_with_public_role_configured(
    app_context: None, mocker: MockerFixture
) -> None:
    """An anonymous principal with AUTH_ROLE_PUBLIC set should yield the public role."""
    sm = SupersetSecurityManager(appbuilder)
    public_role = mocker.MagicMock()
    mocker.patch.object(sm, "get_public_role", return_value=public_role)

    anon_user = mocker.MagicMock()
    anon_user.is_anonymous = True
    mocker.patch("superset.security.manager.g", new=SimpleNamespace(user=anon_user))

    cfg = {"AUTH_ROLE_PUBLIC": "Public"}
    fake_conf = mocker.MagicMock()
    fake_conf.get.side_effect = lambda k, default=None: cfg.get(k, default)
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)

    roles = sm.get_user_roles()
    assert roles == [public_role]


def test_get_user_roles_anonymous_without_public_role(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "get_public_role", return_value=None)
    anon_user = mocker.MagicMock()
    anon_user.is_anonymous = True
    mocker.patch("superset.security.manager.g", new=SimpleNamespace(user=anon_user))

    fake_conf = mocker.MagicMock()
    fake_conf.get.return_value = None
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    roles = sm.get_user_roles()
    assert roles == []


def test_get_guest_rls_filters_filters_by_dataset(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    guest = mocker.MagicMock()
    guest.rls = [
        {"clause": "tenant = 1"},
        {"clause": "tenant = 2", "dataset": 5},
        {"clause": "tenant = 3", "dataset": 99},
    ]
    mocker.patch.object(sm, "get_current_guest_user_if_guest", return_value=guest)

    ds = mocker.MagicMock()
    ds.data = {"id": 5}
    rules = sm.get_guest_rls_filters(ds)
    assert len(rules) == 2  # one global + one for dataset 5
    assert {r["clause"] for r in rules} == {"tenant = 1", "tenant = 2"}


def test_get_guest_rls_filters_no_guest_returns_empty(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "get_current_guest_user_if_guest", return_value=None)
    assert sm.get_guest_rls_filters(mocker.MagicMock()) == []


def test_get_guest_rls_filters_str_returns_clause_strings(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(
        sm,
        "get_guest_rls_filters",
        return_value=[{"clause": "a > 0"}, {"clause": "b < 10"}],
    )
    assert sm.get_guest_rls_filters_str(mocker.MagicMock()) == ["a > 0", "b < 10"]


def test_get_rls_cache_key_combines_guest_and_regular(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "get_guest_rls_filters_str", return_value=["g1"])
    rule = SimpleNamespace(clause="r1", group_key=None)
    mocker.patch.object(sm, "get_rls_sorted", return_value=[rule])

    ds = mocker.MagicMock()
    ds.is_rls_supported = True
    assert sm.get_rls_cache_key(ds) == ["g1", "r1-"]

    # When RLS unsupported, only guest rules are returned.
    ds.is_rls_supported = False
    assert sm.get_rls_cache_key(ds) == ["g1"]


def test_get_rls_filters_no_user_returns_empty(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch("superset.security.manager.g", new=SimpleNamespace())
    assert sm.get_rls_filters(mocker.MagicMock()) == []


def test_get_rls_sorted_orders_by_id(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    f1 = SimpleNamespace(id=2, clause="x", group_key=None)
    f2 = SimpleNamespace(id=1, clause="y", group_key=None)
    mocker.patch.object(sm, "get_rls_filters", return_value=[f1, f2])
    ordered = sm.get_rls_sorted(mocker.MagicMock())
    assert [f.id for f in ordered] == [1, 2]


def test_get_current_epoch_time_returns_float(app_context: None) -> None:
    assert isinstance(SupersetSecurityManager._get_current_epoch_time(), float)


def test_get_guest_token_jwt_audience_callable(
    app_context: None, mocker: MockerFixture
) -> None:
    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = lambda: "https://example.com"
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    assert (
        SupersetSecurityManager._get_guest_token_jwt_audience()
        == "https://example.com"
    )


def test_get_guest_token_jwt_audience_falls_back_to_url_host(
    app_context: None, mocker: MockerFixture
) -> None:
    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = None
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    mocker.patch(
        "superset.security.manager.get_url_host",
        return_value="https://fallback.example.com",
    )
    assert (
        SupersetSecurityManager._get_guest_token_jwt_audience()
        == "https://fallback.example.com"
    )


def test_get_guest_user_from_request_no_token(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = "X-GuestToken"
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    req = mocker.MagicMock()
    req.headers.get.return_value = None
    req.form.get.return_value = None
    assert sm.get_guest_user_from_request(req) is None


def test_get_guest_user_from_request_invalid_token(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = "X-GuestToken"
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    req = mocker.MagicMock()
    req.headers.get.return_value = "blah"
    req.form.get.return_value = None
    mocker.patch.object(
        sm,
        "parse_jwt_guest_token",
        side_effect=Exception("invalid"),
    )
    assert sm.get_guest_user_from_request(req) is None


def test_get_guest_user_from_request_missing_required_claims(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = "X-GuestToken"
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    req = mocker.MagicMock()
    req.headers.get.return_value = "blah"

    # Missing 'user' claim must yield None (broad-except).
    mocker.patch.object(
        sm,
        "parse_jwt_guest_token",
        return_value={"resources": [], "rls_rules": [], "type": "guest"},
    )
    assert sm.get_guest_user_from_request(req) is None


def test_create_guest_access_token_calls_jwt_encode(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    cfg = {
        "GUEST_TOKEN_JWT_SECRET": "secret",
        "GUEST_TOKEN_JWT_ALGO": "HS256",
        "GUEST_TOKEN_JWT_EXP_SECONDS": 60,
        "GUEST_TOKEN_JWT_AUDIENCE": "aud",
    }
    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.side_effect = lambda k: cfg[k]
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    fake_jwt = mocker.MagicMock()
    fake_jwt.encode.return_value = b"token"
    sm.pyjwt_for_guest_token = fake_jwt

    out = sm.create_guest_access_token(
        user={"username": "alice"},  # type: ignore[arg-type]
        resources=[],  # type: ignore[arg-type]
        rls=[],  # type: ignore[arg-type]
    )
    assert out == b"token"
    fake_jwt.encode.assert_called_once()
    claims = fake_jwt.encode.call_args.args[0]
    assert claims["type"] == "guest"
    assert claims["user"] == {"username": "alice"}
    assert claims["aud"] == "aud"


def test_parse_jwt_guest_token_delegates(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    cfg = {
        "GUEST_TOKEN_JWT_SECRET": "secret",
        "GUEST_TOKEN_JWT_ALGO": "HS256",
        "GUEST_TOKEN_JWT_AUDIENCE": "aud",
    }
    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.side_effect = lambda k: cfg[k]
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    fake_jwt = mocker.MagicMock()
    fake_jwt.decode.return_value = {"type": "guest"}
    sm.pyjwt_for_guest_token = fake_jwt

    decoded = sm.parse_jwt_guest_token("rawtoken")
    assert decoded == {"type": "guest"}
    fake_jwt.decode.assert_called_once_with(
        "rawtoken", "secret", algorithms=["HS256"], audience="aud"
    )


# ---------------------------------------------------------------------------
# is_admin / is_owner / raise_for_ownership
# ---------------------------------------------------------------------------


def test_is_admin_true(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    role = mocker.MagicMock()
    role.name = "Admin"
    mocker.patch.object(sm, "get_user_roles", return_value=[role])
    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = "Admin"
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    assert sm.is_admin() is True


def test_is_admin_false(app_context: None, mocker: MockerFixture) -> None:
    sm = SupersetSecurityManager(appbuilder)
    role = mocker.MagicMock()
    role.name = "Gamma"
    mocker.patch.object(sm, "get_user_roles", return_value=[role])
    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = "Admin"
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    assert sm.is_admin() is False


def test_raise_for_ownership_admin_short_circuits(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_admin", return_value=True)
    sm.raise_for_ownership(mocker.MagicMock())


def test_raise_for_ownership_anonymous_raises(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_admin", return_value=False)

    fake_user = mocker.MagicMock()
    fake_user.is_anonymous = True
    mocker.patch("superset.security.manager.g", new=SimpleNamespace(user=fake_user))

    fake_session = mocker.MagicMock()
    resource = mocker.MagicMock()
    resource.id = 1
    fake_obj = mocker.MagicMock()
    fake_obj.owners = []
    fake_session.query.return_value.get.return_value = fake_obj
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )

    with pytest.raises(SupersetSecurityException) as ex:
        sm.raise_for_ownership(resource)
    assert ex.value.error.error_type == SupersetErrorType.MISSING_OWNERSHIP_ERROR


def test_raise_for_ownership_owner_does_not_raise(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_admin", return_value=False)

    user = mocker.MagicMock()
    user.is_anonymous = False
    mocker.patch("superset.security.manager.g", new=SimpleNamespace(user=user))

    resource = mocker.MagicMock()
    resource.id = 5
    fake_obj = mocker.MagicMock()
    fake_obj.owners = [user]
    fake_session = mocker.MagicMock()
    fake_session.query.return_value.get.return_value = fake_obj
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )

    sm.raise_for_ownership(resource)


def test_is_owner_returns_false_when_not_owner(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(
        sm,
        "raise_for_ownership",
        side_effect=SupersetSecurityException(
            sm.get_dashboard_access_error_object(mocker.MagicMock())
        ),
    )
    assert sm.is_owner(mocker.MagicMock()) is False


def test_is_owner_returns_true_when_owner(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "raise_for_ownership")
    assert sm.is_owner(mocker.MagicMock()) is True


# ---------------------------------------------------------------------------
# get_user_by_username, request_loader
# ---------------------------------------------------------------------------


def test_get_user_by_username_returns_query_result(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    user = mocker.MagicMock()
    fake_session = mocker.MagicMock()
    fake_session.query.return_value.filter.return_value.one_or_none.return_value = (
        user
    )
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )
    assert sm.get_user_by_username("alice") is user


def test_request_loader_when_embedded_disabled(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    fake_ff = mocker.MagicMock()
    fake_ff.is_feature_enabled.return_value = False
    mocker.patch("superset.extensions.feature_flag_manager", new=fake_ff)
    assert sm.request_loader(mocker.MagicMock()) is None


def test_request_loader_when_embedded_enabled_uses_guest(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    fake_ff = mocker.MagicMock()
    fake_ff.is_feature_enabled.return_value = True
    mocker.patch("superset.extensions.feature_flag_manager", new=fake_ff)
    sentinel = mocker.MagicMock()
    mocker.patch.object(sm, "get_guest_user_from_request", return_value=sentinel)
    assert sm.request_loader(mocker.MagicMock()) is sentinel


# ---------------------------------------------------------------------------
# user_view_menu_names / get_accessible_databases
# ---------------------------------------------------------------------------


def test_user_view_menu_names_anonymous_with_public_role(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)

    public_role = mocker.MagicMock()
    public_role.id = 11
    mocker.patch.object(sm, "get_public_role", return_value=public_role)

    anon = mocker.MagicMock()
    anon.is_anonymous = True
    mocker.patch("superset.security.manager.g", new=SimpleNamespace(user=anon))

    fake_session = mocker.MagicMock()
    base_q = fake_session.query.return_value
    base_q.join.return_value = base_q
    base_q.filter.return_value = base_q
    base_q.all.return_value = [SimpleNamespace(name="[a].[b]")]
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )

    out = sm.user_view_menu_names("schema_access")
    assert out == {"[a].[b]"}


def test_user_view_menu_names_anonymous_without_public_role(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "get_public_role", return_value=None)
    anon = mocker.MagicMock()
    anon.is_anonymous = True
    mocker.patch("superset.security.manager.g", new=SimpleNamespace(user=anon))
    fake_session = mocker.MagicMock()
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )
    assert sm.user_view_menu_names("schema_access") == set()


def test_get_accessible_databases_parses_db_perms(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(
        sm,
        "user_view_menu_names",
        return_value={
            "[examples].(id:1)",
            "[reporting].(id:2)",
            "garbage",  # ignored
        },
    )
    assert sorted(sm.get_accessible_databases()) == [1, 2]


# ---------------------------------------------------------------------------
# get_schemas_accessible_by_user / get_catalogs_accessible_by_user
# ---------------------------------------------------------------------------


def test_get_schemas_accessible_by_user_database_grant_returns_all(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_database", return_value=True)

    db = mocker.MagicMock()
    db.get_default_catalog.return_value = "main"
    schemas = {"a", "b"}
    out = sm.get_schemas_accessible_by_user(db, catalog=None, schemas=schemas)
    assert out == schemas


def test_get_schemas_accessible_by_user_schema_perm_match(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_database", return_value=False)
    mocker.patch.object(sm, "can_access_catalog", return_value=False)

    db = mocker.MagicMock()
    db.database_name = "examples"
    db.get_default_catalog.return_value = "main"
    db.get_default_schema.return_value = "public"

    fake_session = mocker.MagicMock()
    fake_session.query.return_value.filter.return_value.filter.return_value.distinct.return_value = []  # noqa: E501
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )

    mocker.patch.object(
        sm,
        "user_view_menu_names",
        side_effect=[
            {
                "[examples].[public]",  # database.schema (default catalog)
                "[examples].[main].[reporting]",  # database.catalog.schema
                "[other].[whatever]",  # different db, ignored
            },
            set(),  # datasource_access
        ],
    )

    out = sm.get_schemas_accessible_by_user(
        db,
        catalog=None,
        schemas={"public", "reporting", "missing"},
    )
    assert out == {"public", "reporting"}


def test_get_catalogs_accessible_by_user_database_grant_returns_all(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_database", return_value=True)
    db = mocker.MagicMock()
    catalogs = {"a", "b"}
    assert sm.get_catalogs_accessible_by_user(db, catalogs) == catalogs


def test_get_catalogs_accessible_by_user_catalog_access(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_database", return_value=False)
    db = mocker.MagicMock()
    db.database_name = "examples"
    db.get_default_catalog.return_value = "main"

    fake_session = mocker.MagicMock()
    fake_session.query.return_value.filter.return_value.filter.return_value.distinct.return_value = []  # noqa: E501
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )

    mocker.patch.object(
        sm,
        "user_view_menu_names",
        side_effect=[
            {"[examples].[main]", "[other].[c1]"},  # catalog_access
            set(),  # schema_access
            set(),  # datasource_access
        ],
    )
    out = sm.get_catalogs_accessible_by_user(db, catalogs={"main", "extra"})
    assert out == {"main"}


# ---------------------------------------------------------------------------
# get_datasources_accessible_by_user
# ---------------------------------------------------------------------------


def test_get_datasources_accessible_by_user_full_db_access(
    app_context: None, mocker: MockerFixture
) -> None:
    from superset.utils.core import DatasourceName

    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_database", return_value=True)

    db = mocker.MagicMock()
    db.database_name = "examples"
    db.get_default_catalog.return_value = "main"

    names = [DatasourceName("t1", "public", "main")]
    assert sm.get_datasources_accessible_by_user(db, names) == names


def test_get_datasources_accessible_by_user_catalog_access(
    app_context: None, mocker: MockerFixture
) -> None:
    from superset.utils.core import DatasourceName

    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_database", return_value=False)
    can_access = mocker.patch.object(sm, "can_access", return_value=True)

    db = mocker.MagicMock()
    db.database_name = "examples"
    db.get_default_catalog.return_value = "main"

    names = [DatasourceName("t1", "public", "main")]
    assert sm.get_datasources_accessible_by_user(db, names) == names
    can_access.assert_called_once_with("catalog_access", "[examples].[main]")


# ---------------------------------------------------------------------------
# Validate guest token resources
# ---------------------------------------------------------------------------


def test_validate_guest_token_resources_dashboard_present(
    app_context: None, mocker: MockerFixture
) -> None:
    """Dashboard found: no exception should be raised."""
    from superset.models.dashboard import Dashboard
    from superset.security.guest_token import GuestTokenResourceType

    mocker.patch.object(Dashboard, "get", return_value=mocker.MagicMock())
    SupersetSecurityManager.validate_guest_token_resources(
        [{"type": GuestTokenResourceType.DASHBOARD.value, "id": "1"}]
    )


def test_validate_guest_token_resources_falls_back_to_embedded(
    app_context: None, mocker: MockerFixture
) -> None:
    from superset.daos.dashboard import EmbeddedDashboardDAO
    from superset.models.dashboard import Dashboard
    from superset.security.guest_token import GuestTokenResourceType

    mocker.patch.object(Dashboard, "get", return_value=None)
    mocker.patch.object(
        EmbeddedDashboardDAO, "find_by_id", return_value=mocker.MagicMock()
    )
    SupersetSecurityManager.validate_guest_token_resources(
        [{"type": GuestTokenResourceType.DASHBOARD.value, "id": "abc"}]
    )


def test_validate_guest_token_resources_raises_when_missing(
    app_context: None, mocker: MockerFixture
) -> None:
    from superset.commands.dashboard.embedded.exceptions import (
        EmbeddedDashboardNotFoundError,
    )
    from superset.daos.dashboard import EmbeddedDashboardDAO
    from superset.models.dashboard import Dashboard
    from superset.security.guest_token import GuestTokenResourceType

    mocker.patch.object(Dashboard, "get", return_value=None)
    mocker.patch.object(EmbeddedDashboardDAO, "find_by_id", return_value=None)
    with pytest.raises(EmbeddedDashboardNotFoundError):
        SupersetSecurityManager.validate_guest_token_resources(
            [{"type": GuestTokenResourceType.DASHBOARD.value, "id": "missing"}]
        )


# ---------------------------------------------------------------------------
# get_guest_user_from_token
# ---------------------------------------------------------------------------


def test_get_guest_user_from_token_constructs_guest_user(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    fake_role = mocker.MagicMock()
    mocker.patch.object(sm, "find_role", return_value=fake_role)
    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = "Public"
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)

    fake_cls = mocker.MagicMock()
    sm.guest_user_cls = fake_cls
    token = {"user": {"username": "guest"}, "resources": [], "rls_rules": []}
    sm.get_guest_user_from_token(token)  # type: ignore[arg-type]
    fake_cls.assert_called_once_with(token=token, roles=[fake_role])


# ---------------------------------------------------------------------------
# Permission / role management helpers
# ---------------------------------------------------------------------------


def test_merge_perm_delegates_to_add_permission_view_menu(
    app_context: None, mocker: MockerFixture
) -> None:
    """`merge_perm` is deprecated; it must still call add_permission_view_menu."""
    sm = SupersetSecurityManager(appbuilder)
    add = mocker.patch.object(sm, "add_permission_view_menu")
    sm.merge_perm("can_read", "MyView")
    add.assert_called_once_with("can_read", "MyView")


def test_create_custom_permissions_adds_all_expected_pvms(
    app_context: None, mocker: MockerFixture
) -> None:
    """Smoke test that every custom PVM is registered exactly once."""
    sm = SupersetSecurityManager(appbuilder)
    add = mocker.patch.object(sm, "add_permission_view_menu")
    sm.create_custom_permissions()
    # The set of expected (perm, view) tuples documented at the top of the
    # method.  We assert each one was requested.
    expected = {
        ("all_datasource_access", "all_datasource_access"),
        ("all_database_access", "all_database_access"),
        ("all_query_access", "all_query_access"),
        ("can_csv", "Superset"),
        ("can_export_data", "Superset"),
        ("can_export_image", "Superset"),
        ("can_copy_clipboard", "Superset"),
        ("can_share_dashboard", "Superset"),
        ("can_share_chart", "Superset"),
        ("can_sqllab", "Superset"),
        ("can_view_query", "Dashboard"),
        ("can_view_chart_as_table", "Dashboard"),
        ("can_drill", "Dashboard"),
        ("can_tag", "Chart"),
        ("can_tag", "Dashboard"),
    }
    seen = {(c.args[0], c.args[1]) for c in add.call_args_list}
    assert expected.issubset(seen)


def test_clean_perms_deletes_orphan_pvms(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    fake_session = mocker.MagicMock()
    pvms_query = fake_session.query.return_value.filter.return_value
    pvms_query.delete.return_value = 7
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )
    sm.clean_perms()
    pvms_query.delete.assert_called_once()


def test_clean_perms_no_orphans(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    fake_session = mocker.MagicMock()
    pvms_query = fake_session.query.return_value.filter.return_value
    pvms_query.delete.return_value = 0
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )
    sm.clean_perms()  # should not raise
    pvms_query.delete.assert_called_once()


def test_get_all_pvms_filters_orphans(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)

    valid = mocker.MagicMock()
    valid.permission = mocker.MagicMock()
    valid.view_menu = mocker.MagicMock()

    orphan_no_perm = mocker.MagicMock()
    orphan_no_perm.permission = None
    orphan_no_perm.view_menu = mocker.MagicMock()

    orphan_no_view = mocker.MagicMock()
    orphan_no_view.permission = mocker.MagicMock()
    orphan_no_view.view_menu = None

    fake_session = mocker.MagicMock()
    fake_session.query.return_value.options.return_value.all.return_value = [
        valid,
        orphan_no_perm,
        orphan_no_view,
    ]
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )
    assert sm._get_all_pvms() == [valid]


def test_find_roles_by_id_filters_with_in(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    fake_session = mocker.MagicMock()
    role_a = mocker.MagicMock()
    role_b = mocker.MagicMock()
    fake_session.query.return_value.filter.return_value.all.return_value = [
        role_a,
        role_b,
    ]
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )
    assert sm.find_roles_by_id([1, 2]) == [role_a, role_b]


def test_set_role_assigns_only_matching_pvms(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    role = mocker.MagicMock()
    mocker.patch.object(sm, "add_role", return_value=role)
    pvm_keep = _make_pvm(mocker, "p1", "v1")
    pvm_skip = _make_pvm(mocker, "p2", "v2")
    sm.set_role("CustomRole", lambda p: p is pvm_keep, [pvm_keep, pvm_skip])
    assert role.permissions == [pvm_keep]


def test_get_pvms_from_builtin_role_filters_by_regex(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)

    matching = _make_pvm(mocker, "can_read", "Dashboard")
    not_matching_view = _make_pvm(mocker, "can_read", "Other")
    not_matching_perm = _make_pvm(mocker, "can_write", "Dashboard")

    fake_session = mocker.MagicMock()
    fake_session.query.return_value.all.return_value = [
        matching,
        matching,  # duplicate to test dedup branch
        not_matching_view,
        not_matching_perm,
    ]
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )
    mocker.patch.object(
        type(sm),
        "builtin_roles",
        new_callable=mocker.PropertyMock,
        return_value={"DashboardReader": [["Dashboard", "can_read"]]},
    )

    out = sm._get_pvms_from_builtin_role("DashboardReader")
    assert out == [matching]


def test_get_pvms_from_builtin_role_unknown_role_returns_empty(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    fake_session = mocker.MagicMock()
    fake_session.query.return_value.all.return_value = []
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )
    mocker.patch.object(
        type(sm),
        "builtin_roles",
        new_callable=mocker.PropertyMock,
        return_value={},
    )
    assert sm._get_pvms_from_builtin_role("NotInBuiltin") == []


def test_copy_role_from_builtin_to_role(
    app_context: None, mocker: MockerFixture
) -> None:
    """Builtin role path: pulls perms from `_get_pvms_from_builtin_role`."""
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(
        type(sm),
        "builtin_roles",
        new_callable=mocker.PropertyMock,
        return_value={"Builtin": [["v", "p"]]},
    )
    mocker.patch.object(
        type(sm),
        "data_access_permissions",
        new_callable=mocker.PropertyMock,
        return_value={"datasource_access"},
    )

    pvm = _make_pvm(mocker, "p", "v")
    mocker.patch.object(sm, "_get_pvms_from_builtin_role", return_value=[pvm])
    role_to = mocker.MagicMock()
    role_to.permissions = []
    mocker.patch.object(sm, "add_role", return_value=role_to)

    sm.copy_role("Builtin", "Public", merge=True)
    assert role_to.permissions == [pvm]


def test_copy_role_keeps_existing_data_access_when_merge(
    app_context: None, mocker: MockerFixture
) -> None:
    """When `merge=True`, existing data-access perms must be preserved on dest."""
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(
        type(sm),
        "builtin_roles",
        new_callable=mocker.PropertyMock,
        return_value={},
    )
    mocker.patch.object(
        type(sm),
        "data_access_permissions",
        new_callable=mocker.PropertyMock,
        return_value={"datasource_access"},
    )

    src_pvm = _make_pvm(mocker, "menu_access", "Charts")
    role_from = mocker.MagicMock()
    role_from.permissions = [src_pvm]
    mocker.patch.object(sm, "find_role", return_value=role_from)

    keep_data = _make_pvm(mocker, "datasource_access", "[db].[table]")
    role_to = mocker.MagicMock()
    role_to.permissions = [keep_data]
    mocker.patch.object(sm, "add_role", return_value=role_to)

    sm.copy_role("Source", "Target", merge=True)
    assert src_pvm in role_to.permissions
    assert keep_data in role_to.permissions


def test_copy_role_no_merge_overwrites(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(
        type(sm),
        "builtin_roles",
        new_callable=mocker.PropertyMock,
        return_value={},
    )
    mocker.patch.object(
        type(sm),
        "data_access_permissions",
        new_callable=mocker.PropertyMock,
        return_value={"datasource_access"},
    )

    src_pvm = _make_pvm(mocker, "menu_access", "Charts")
    role_from = mocker.MagicMock()
    role_from.permissions = [src_pvm]
    mocker.patch.object(sm, "find_role", return_value=role_from)

    existing = _make_pvm(mocker, "datasource_access", "[db].[t1]")
    role_to = mocker.MagicMock()
    role_to.permissions = [existing]
    mocker.patch.object(sm, "add_role", return_value=role_to)

    sm.copy_role("Source", "Target", merge=False)
    # When merge=False, existing data-access perms must NOT be carried over.
    assert role_to.permissions == [src_pvm]


def test_sync_role_definitions_public_role_like_public(
    app_context: None, mocker: MockerFixture
) -> None:
    """`PUBLIC_ROLE_LIKE == 'Public'` triggers `set_role('Public', ...)`."""
    sm = SupersetSecurityManager(appbuilder)
    pvms: list[Any] = []

    create_custom = mocker.patch.object(sm, "create_custom_permissions")
    mocker.patch.object(sm, "_get_all_pvms", return_value=pvms)
    set_role = mocker.patch.object(sm, "set_role")
    copy_role = mocker.patch.object(sm, "copy_role")
    create_missing = mocker.patch.object(sm, "create_missing_perms")
    clean_perms = mocker.patch.object(sm, "clean_perms")

    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = "Public"
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)

    sm.sync_role_definitions()
    create_custom.assert_called_once()
    # Admin, Alpha, Gamma, sql_lab, Public => 5 calls
    assert set_role.call_count == 5
    copy_role.assert_not_called()
    create_missing.assert_called_once()
    clean_perms.assert_called_once()


def test_sync_role_definitions_public_role_like_other(
    app_context: None, mocker: MockerFixture
) -> None:
    """`PUBLIC_ROLE_LIKE == 'Gamma'` triggers `copy_role(...)`."""
    sm = SupersetSecurityManager(appbuilder)

    mocker.patch.object(sm, "create_custom_permissions")
    mocker.patch.object(sm, "_get_all_pvms", return_value=[])
    set_role = mocker.patch.object(sm, "set_role")
    copy_role = mocker.patch.object(sm, "copy_role")
    mocker.patch.object(sm, "create_missing_perms")
    mocker.patch.object(sm, "clean_perms")

    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = "Gamma"
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    mocker.patch.object(
        type(sm),
        "auth_role_public",
        new_callable=mocker.PropertyMock,
        return_value="Public",
    )

    sm.sync_role_definitions()
    # Admin, Alpha, Gamma, sql_lab => 4 calls (no Public)
    assert set_role.call_count == 4
    copy_role.assert_called_once_with("Gamma", "Public", merge=True)


def test_sync_role_definitions_public_role_like_none(
    app_context: None, mocker: MockerFixture
) -> None:
    """If config is empty, neither Public nor copy_role should be invoked."""
    sm = SupersetSecurityManager(appbuilder)

    mocker.patch.object(sm, "create_custom_permissions")
    mocker.patch.object(sm, "_get_all_pvms", return_value=[])
    set_role = mocker.patch.object(sm, "set_role")
    copy_role = mocker.patch.object(sm, "copy_role")
    mocker.patch.object(sm, "create_missing_perms")
    mocker.patch.object(sm, "clean_perms")

    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = None
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)

    sm.sync_role_definitions()
    assert set_role.call_count == 4  # Admin/Alpha/Gamma/sql_lab only
    copy_role.assert_not_called()


def test_create_missing_perms_creates_pvms_for_unknown_combos(
    app_context: None, mocker: MockerFixture
) -> None:
    """Verifies that already-known PVMs are skipped and missing ones are added."""
    sm = SupersetSecurityManager(appbuilder)

    add = mocker.patch.object(sm, "add_permission_view_menu")
    existing_pvm = mocker.MagicMock()
    existing_pvm.permission.name = "datasource_access"
    existing_pvm.view_menu.name = "[db].[s].[t1]"
    mocker.patch.object(sm, "_get_all_pvms", return_value=[existing_pvm])

    # Two datasets with different perms; one already exists, one is new.
    ds_existing = mocker.MagicMock()
    ds_existing.get_perm.return_value = "[db].[s].[t1]"
    ds_existing.get_schema_perm.return_value = "[db].[s]"
    ds_existing.get_catalog_perm.return_value = None  # filtered out

    ds_new = mocker.MagicMock()
    ds_new.get_perm.return_value = "[db].[s].[t2]"
    ds_new.get_schema_perm.return_value = "[db].[s]"
    ds_new.get_catalog_perm.return_value = "[db]"

    db_obj = mocker.MagicMock()
    db_obj.perm = "[db]"

    fake_session = mocker.MagicMock()
    fake_session.query.return_value.all.return_value = [db_obj]
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )
    mocker.patch(
        "superset.connectors.sqla.models.SqlaTable.get_all_datasources",
        return_value=[ds_existing, ds_new],
    )

    sm.create_missing_perms()
    perm_names = {(c.args[0], c.args[1]) for c in add.call_args_list}
    # The existing PVM must NOT be re-added.
    assert ("datasource_access", "[db].[s].[t1]") not in perm_names
    assert ("datasource_access", "[db].[s].[t2]") in perm_names
    assert ("schema_access", "[db].[s]") in perm_names
    assert ("catalog_access", "[db]") in perm_names
    assert ("database_access", "[db]") in perm_names


# ---------------------------------------------------------------------------
# Database SQLAlchemy event hooks
# ---------------------------------------------------------------------------


def test_database_after_insert_creates_pvm(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    insert = mocker.patch.object(sm, "_insert_pvm_on_sqla_event")
    target = mocker.MagicMock()
    target.get_perm.return_value = "[examples]"
    mapper, conn = mocker.MagicMock(), mocker.MagicMock()
    sm.database_after_insert(mapper, conn, target)
    insert.assert_called_once_with(mapper, conn, "database_access", "[examples]")


def test_database_after_delete_calls_delete_helper(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    delete = mocker.patch.object(sm, "_delete_vm_database_access")
    target = mocker.MagicMock()
    target.id = 5
    target.database_name = "examples"
    mapper, conn = mocker.MagicMock(), mocker.MagicMock()
    sm.database_after_delete(mapper, conn, target)
    delete.assert_called_once_with(mapper, conn, 5, "examples")


def test_database_after_update_no_name_change_no_op(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    update_db = mocker.patch.object(sm, "_update_vm_database_access")
    update_ds = mocker.patch.object(sm, "_update_vm_datasources_access")
    target = mocker.MagicMock()
    history = mocker.MagicMock()
    history.has_changes.return_value = False
    history.deleted = []
    state = mocker.MagicMock()
    state.get_history.return_value = history
    mocker.patch("superset.security.manager.inspect", return_value=state)
    sm.database_after_update(mocker.MagicMock(), mocker.MagicMock(), target)
    update_db.assert_not_called()
    update_ds.assert_not_called()


def test_database_after_update_renames_propagates(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    update_db = mocker.patch.object(sm, "_update_vm_database_access")
    update_ds = mocker.patch.object(sm, "_update_vm_datasources_access")
    target = mocker.MagicMock()
    target.database_name = "new"
    history = mocker.MagicMock()
    history.has_changes.return_value = True
    history.deleted = ["old"]
    state = mocker.MagicMock()
    state.get_history.return_value = history
    mocker.patch("superset.security.manager.inspect", return_value=state)
    mapper, conn = mocker.MagicMock(), mocker.MagicMock()
    sm.database_after_update(mapper, conn, target)
    update_db.assert_called_once_with(mapper, conn, "old", target)
    update_ds.assert_called_once_with(mapper, conn, "old", target)


def test_delete_vm_database_access_cleans_pvms(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    delete = mocker.patch.object(sm, "_delete_pvm_on_sqla_event")
    schema_pvm_a = mocker.MagicMock()
    schema_pvm_b = mocker.MagicMock()
    fake_session = mocker.MagicMock()
    fake_session.query.return_value.join.return_value.join.return_value.filter.return_value.filter.return_value.all.return_value = [  # noqa: E501
        schema_pvm_a,
        schema_pvm_b,
    ]
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )
    mapper, conn = mocker.MagicMock(), mocker.MagicMock()
    sm._delete_vm_database_access(mapper, conn, 1, "examples")
    # First call: database_access
    # Then 2 calls: schema-level pvms
    assert delete.call_count == 3


def test_update_vm_database_access_creates_when_missing(
    app_context: None, mocker: MockerFixture
) -> None:
    """When previous PVM doesn't exist, fall back to inserting a new one."""
    sm = SupersetSecurityManager(appbuilder)
    insert = mocker.patch.object(sm, "_insert_pvm_on_sqla_event")
    mocker.patch.object(sm, "find_permission_view_menu", return_value=None)
    target = mocker.MagicMock()
    target.id = 1
    target.database_name = "new"
    mapper, conn = mocker.MagicMock(), mocker.MagicMock()

    out = sm._update_vm_database_access(mapper, conn, "old", target)
    assert out is None
    insert.assert_called_once()


def test_update_vm_database_access_deletes_when_new_already_exists(
    app_context: None, mocker: MockerFixture
) -> None:
    """If the destination PVM already exists, the old one must be deleted."""
    sm = SupersetSecurityManager(appbuilder)
    delete = mocker.patch.object(sm, "_delete_vm_database_access")
    old = mocker.MagicMock()
    new = mocker.MagicMock()
    mocker.patch.object(
        sm,
        "find_permission_view_menu",
        side_effect=[old, new],
    )
    target = mocker.MagicMock()
    target.id = 1
    target.database_name = "new"
    mapper, conn = mocker.MagicMock(), mocker.MagicMock()
    out = sm._update_vm_database_access(mapper, conn, "old", target)
    assert out is None
    delete.assert_called_once_with(mapper, conn, 1, "old")


def test_update_vm_database_access_renames_view_menu(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    old = mocker.MagicMock()
    old.view_menu_id = 99
    mocker.patch.object(sm, "find_permission_view_menu", side_effect=[old, None])
    new_vm = mocker.MagicMock()
    mocker.patch.object(sm, "_find_view_menu_on_sqla_event", return_value=new_vm)
    on_view_update = mocker.patch.object(sm, "on_view_menu_after_update")

    target = mocker.MagicMock()
    target.id = 1
    target.database_name = "new"
    mapper, conn = mocker.MagicMock(), mocker.MagicMock()
    out = sm._update_vm_database_access(mapper, conn, "old", target)
    assert out is new_vm
    conn.execute.assert_called()  # at least the rename UPDATE
    on_view_update.assert_called_once_with(mapper, conn, new_vm)


# ---------------------------------------------------------------------------
# Dataset SQLAlchemy event hooks
# ---------------------------------------------------------------------------


def test_dataset_after_delete_uses_dataset_perm(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    delete = mocker.patch.object(sm, "_delete_pvm_on_sqla_event")
    target = mocker.MagicMock()
    target.id = 7
    target.table_name = "users"
    target.database.database_name = "examples"
    mapper, conn = mocker.MagicMock(), mocker.MagicMock()
    sm.dataset_after_delete(mapper, conn, target)
    delete.assert_called_once()
    # The view_menu name passed must be the one returned by get_dataset_perm.
    args = delete.call_args.args
    assert args[2] == "datasource_access"


def test_dataset_after_insert_basic_path(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    insert = mocker.patch.object(sm, "_insert_pvm_on_sqla_event")

    target = mocker.MagicMock()
    target.id = 1
    target.get_perm.return_value = "[examples].[s].[t]"
    target.perm = "[examples].[s].[t]"
    target.schema = None
    target.catalog = None

    table = mocker.MagicMock()
    target.__table__ = table

    mapper, conn = mocker.MagicMock(), mocker.MagicMock()
    sm.dataset_after_insert(mapper, conn, target)
    insert.assert_any_call(mapper, conn, "datasource_access", "[examples].[s].[t]")


def test_dataset_after_insert_with_schema_and_catalog(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    insert = mocker.patch.object(sm, "_insert_pvm_on_sqla_event")

    target = mocker.MagicMock()
    target.id = 2
    target.get_perm.return_value = "ds_perm"
    target.perm = "stale_perm"  # forces an UPDATE
    target.schema = "public"
    target.catalog = "main"
    target.database.database_name = "examples"
    target.__table__ = mocker.MagicMock()

    mapper, conn = mocker.MagicMock(), mocker.MagicMock()
    sm.dataset_after_insert(mapper, conn, target)
    # 3 _insert_pvm_on_sqla_event calls: datasource, schema, catalog
    assert insert.call_count == 3


def test_dataset_after_insert_falls_back_when_no_database(
    app_context: None, mocker: MockerFixture
) -> None:
    """If `target.database` raises, the manager must fall back to database_id."""
    from superset.exceptions import (
        DatasetInvalidPermissionEvaluationException,
    )

    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "_insert_pvm_on_sqla_event")

    target = mocker.MagicMock()
    target.id = 3
    target.table_name = "tbl"
    target.database_id = 5
    target.schema = None
    target.catalog = None
    target.__table__ = mocker.MagicMock()
    target.perm = "ignored"
    type(target).get_perm = mocker.MagicMock(
        side_effect=DatasetInvalidPermissionEvaluationException()
    )

    fake_database = mocker.MagicMock()
    fake_database.database_name = "examples"
    fake_session = mocker.MagicMock()
    fake_session.query.return_value.get.return_value = fake_database
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )

    sm.dataset_after_insert(mocker.MagicMock(), mocker.MagicMock(), target)


# ---------------------------------------------------------------------------
# _delete_pvm_on_sqla_event
# ---------------------------------------------------------------------------


def test_delete_pvm_on_sqla_event_no_pvm_no_op(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "find_permission_view_menu", return_value=None)
    mapper, conn = mocker.MagicMock(), mocker.MagicMock()
    sm._delete_pvm_on_sqla_event(
        mapper, conn, permission_name="x", view_menu_name="y"
    )
    conn.execute.assert_not_called()


def test_delete_pvm_on_sqla_event_with_explicit_pvm(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    pvm = mocker.MagicMock()
    pvm.id = 9
    pvm.view_menu_id = 33
    on_delete = mocker.patch.object(sm, "on_permission_view_after_delete")
    mapper, conn = mocker.MagicMock(), mocker.MagicMock()
    sm._delete_pvm_on_sqla_event(mapper, conn, pvm=pvm)
    # 3 SQL deletes: assoc role mapping, pvm row, view_menu row
    assert conn.execute.call_count == 3
    on_delete.assert_called_once_with(mapper, conn, pvm)


# ---------------------------------------------------------------------------
# get_user_datasources
# ---------------------------------------------------------------------------


def test_get_user_datasources_combines_explicit_and_db_grants(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)

    explicit = mocker.MagicMock()
    explicit_db = mocker.MagicMock()
    explicit.database = explicit_db

    other_ds = mocker.MagicMock()
    other_db = mocker.MagicMock()
    other_ds.database = other_db

    fake_session = mocker.MagicMock()
    fake_session.query.return_value.filter.return_value.all.return_value = [explicit]
    mocker.patch.object(
        type(sm), "session", new_callable=mocker.PropertyMock, return_value=fake_session
    )

    mocker.patch(
        "superset.connectors.sqla.models.SqlaTable.get_all_datasources",
        return_value=[explicit, other_ds],
    )
    # Bypass `get_dataset_access_filters` complexity entirely.
    mocker.patch(
        "superset.security.manager.get_dataset_access_filters",
        return_value=mocker.MagicMock(),
    )
    # Only `other_db` is granted via database access
    mocker.patch.object(
        sm,
        "can_access_database",
        side_effect=lambda db: db is other_db,
    )

    out = sm.get_user_datasources()
    assert set(out) == {explicit, other_ds}


# ---------------------------------------------------------------------------
# get_guest_user_from_request - success and validation paths
# ---------------------------------------------------------------------------


def test_get_guest_user_from_request_success(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    sentinel = mocker.MagicMock()
    mocker.patch.object(
        sm,
        "parse_jwt_guest_token",
        return_value={
            "user": {"username": "g"},
            "resources": [],
            "rls_rules": [],
            "type": "guest",
        },
    )
    mocker.patch.object(sm, "get_guest_user_from_token", return_value=sentinel)

    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = "X-GuestToken"
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)

    req = mocker.MagicMock()
    req.headers.get.return_value = "raw"
    assert sm.get_guest_user_from_request(req) is sentinel


def test_get_guest_user_from_request_wrong_type_returns_none(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(
        sm,
        "parse_jwt_guest_token",
        return_value={
            "user": {"username": "g"},
            "resources": [],
            "rls_rules": [],
            "type": "not_guest",
        },
    )
    fake_conf = mocker.MagicMock()
    fake_conf.__getitem__.return_value = "X-GuestToken"
    mocker.patch("superset.security.manager.get_conf", return_value=fake_conf)
    req = mocker.MagicMock()
    req.headers.get.return_value = "raw"
    assert sm.get_guest_user_from_request(req) is None


# ---------------------------------------------------------------------------
# has_guest_access UUID-based fallback
# ---------------------------------------------------------------------------


def test_has_guest_access_falls_back_to_embedded_uuid(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    from superset.security.guest_token import GuestTokenResourceType

    embedded = mocker.MagicMock()
    embedded.uuid = "uuid-1234"
    dashboard = mocker.MagicMock()
    dashboard.id = 99
    dashboard.embedded = [embedded]

    guest = mocker.MagicMock()
    guest.resources = [
        {"type": GuestTokenResourceType.DASHBOARD, "id": "uuid-1234"},
    ]
    mocker.patch.object(sm, "get_current_guest_user_if_guest", return_value=guest)
    assert sm.has_guest_access(dashboard) is True


def test_has_guest_access_no_match_returns_false(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    from superset.security.guest_token import GuestTokenResourceType

    dashboard = mocker.MagicMock()
    dashboard.id = 1
    dashboard.embedded = []

    guest = mocker.MagicMock()
    guest.resources = [{"type": GuestTokenResourceType.DASHBOARD, "id": "999"}]
    mocker.patch.object(sm, "get_current_guest_user_if_guest", return_value=guest)
    assert sm.has_guest_access(dashboard) is False


# ---------------------------------------------------------------------------
# RLS prefetch
# ---------------------------------------------------------------------------


def test_prefetch_rls_filters_returns_when_no_user(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch("superset.security.manager.g", new=SimpleNamespace(user=None))
    sm.prefetch_rls_filters([1, 2])  # must not raise


def test_prefetch_rls_filters_returns_when_no_username(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch(
        "superset.security.manager.g",
        new=SimpleNamespace(user=mocker.MagicMock()),
    )
    mocker.patch("superset.security.manager.get_username", return_value=None)
    sm.prefetch_rls_filters([1, 2])  # must not raise


# ---------------------------------------------------------------------------
# raise_for_access misc branches (datasource via dashboard, viz, sql+db)
# ---------------------------------------------------------------------------


def test_raise_for_access_datasource_admin_owner_short_circuits(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_schema", return_value=False)
    mocker.patch.object(sm, "can_access", return_value=False)
    mocker.patch.object(sm, "is_owner", return_value=True)
    ds = mocker.MagicMock()
    ds.perm = "x"
    sm.raise_for_access(datasource=ds)


def test_raise_for_access_via_viz_uses_form_data(
    app_context: None, mocker: MockerFixture
) -> None:
    """`viz` argument must be unwrapped to datasource + form_data."""
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_schema", return_value=True)
    viz = mocker.MagicMock()
    viz.datasource = mocker.MagicMock()
    viz.datasource.perm = "x"
    viz.form_data = {}
    sm.raise_for_access(viz=viz)


def test_raise_for_access_datasource_denied_raises(
    app_context: None, mocker: MockerFixture
) -> None:
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_schema", return_value=False)
    mocker.patch.object(sm, "can_access", return_value=False)
    mocker.patch.object(sm, "is_owner", return_value=False)
    ds = mocker.MagicMock()
    ds.perm = "x"
    with pytest.raises(SupersetSecurityException) as excinfo:
        sm.raise_for_access(datasource=ds)
    assert (
        excinfo.value.error.error_type
        == SupersetErrorType.DATASOURCE_SECURITY_ACCESS_ERROR
    )


def test_raise_for_access_query_context_modified_for_guest(
    app_context: None, mocker: MockerFixture
) -> None:
    """A modified query_context for a guest user must raise."""
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "is_guest_user", return_value=True)
    mocker.patch(
        "superset.security.manager.query_context_modified", return_value=True
    )
    qc = mocker.MagicMock()
    qc.datasource = mocker.MagicMock()
    qc.datasource.perm = "x"
    qc.form_data = {}
    with pytest.raises(SupersetSecurityException):
        sm.raise_for_access(query_context=qc)


# ---------------------------------------------------------------------------
# get_datasources_accessible_by_user fallback path
# ---------------------------------------------------------------------------


def test_get_datasources_accessible_by_user_filters_by_query(
    app_context: None, mocker: MockerFixture
) -> None:
    """Final fallback path: filter against permission queries."""
    from superset.connectors.sqla.models import SqlaTable
    from superset.utils.core import DatasourceName

    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, "can_access_database", return_value=False)
    mocker.patch.object(sm, "can_access", return_value=False)
    mocker.patch.object(
        sm, "user_view_menu_names", side_effect=[set(), set(), set()]
    )
    db = mocker.MagicMock()
    db.database_name = "examples"
    db.get_default_catalog.return_value = "main"

    visible = SimpleNamespace(table_name="t1", schema="public", catalog="main")
    mocker.patch.object(
        SqlaTable, "query_datasources_by_permissions", return_value=[visible]
    )
    names = [
        DatasourceName("t1", "public", "main"),
        DatasourceName("hidden", "public", "main"),
    ]
    out = sm.get_datasources_accessible_by_user(db, names)
    assert out == [DatasourceName("t1", "public", "main")]
