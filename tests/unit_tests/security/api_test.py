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
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from flask_wtf.csrf import generate_csrf
from marshmallow import ValidationError
from pytest_mock import MockerFixture

from superset.commands.dashboard.embedded.exceptions import (
    EmbeddedDashboardNotFoundError,
)
from superset.commands.exceptions import ForbiddenError
from superset.exceptions import SupersetGenericErrorException
from superset.extensions import csrf
from superset.security.api import (
    GuestTokenCreateSchema,
    PermissiveSchema,
    ResourceSchema,
    RlsRuleSchema,
    RoleResponseSchema,
    RolesResponseSchema,
    UserSchema,
    guest_token_create_schema,
)
from superset.security.guest_token import GuestTokenResourceType


@pytest.mark.parametrize(
    "app",
    [{"WTF_CSRF_ENABLED": True}],
    indirect=True,
)
def test_csrf_exempt_blueprints(app_context: None) -> None:
    """
    Test that only FAB security API blueprints (which use token-based auth)
    are exempt from CSRF protection.
    """
    assert {blueprint.name for blueprint in csrf._exempt_blueprints} == {
        "SupersetGroupApi",
        "MenuApi",
        "SecurityApi",
        "OpenApi",
        "PermissionViewMenuApi",
        "SupersetRoleApi",
        "SupersetUserApi",
        "PermissionApi",
        "ViewMenuApi",
    }


@pytest.mark.parametrize(
    "app",
    [
        {
            "WTF_CSRF_ENABLED": True,
            "FAB_API_KEY_ENABLED": True,
        }
    ],
    indirect=True,
)
def test_csrf_exempt_blueprints_with_api_key(app: Any, app_context: None) -> None:
    """
    Test that ApiKeyApi blueprint is CSRF-exempt when FAB_API_KEY_ENABLED
    config is enabled.
    """
    assert "ApiKeyApi" in {blueprint.name for blueprint in csrf._exempt_blueprints}


# ---------------------------------------------------------------------------
# Schema unit tests (Authentication / Guest Token compliance coverage).
#
# These tests exercise the marshmallow schemas exposed by
# ``superset.security.api`` so the OCC-relevant payload validation surface is
# explicitly covered.
# ---------------------------------------------------------------------------


class _ChildSchema(PermissiveSchema):
    """Minimal subclass used to exercise PermissiveSchema's EXCLUDE behavior."""

    pass


def test_permissive_schema_excludes_unknown_fields() -> None:
    """PermissiveSchema must silently drop unknown fields rather than raising."""
    result = _ChildSchema().load({"unexpected": "value", "another": 123})
    assert result == {}


def test_user_schema_loads_known_fields() -> None:
    """UserSchema accepts username/first_name/last_name and ignores extras."""
    payload = {
        "username": "bob",
        "first_name": "Bob",
        "last_name": "Builder",
        "extraneous": "ignored",
    }
    loaded = UserSchema().load(payload)
    assert loaded == {
        "username": "bob",
        "first_name": "Bob",
        "last_name": "Builder",
    }


def test_resource_schema_converts_enum_to_value() -> None:
    """ResourceSchema's post_load hook should reduce the enum to its raw value."""
    loaded = ResourceSchema().load({"type": "dashboard", "id": "abc-123"})
    assert loaded == {"type": "dashboard", "id": "abc-123"}
    assert not isinstance(loaded["type"], GuestTokenResourceType)
    assert isinstance(loaded["type"], str)


def test_resource_schema_rejects_unknown_resource_type() -> None:
    """ResourceSchema must surface a ValidationError for unsupported types."""
    with pytest.raises(ValidationError) as excinfo:
        ResourceSchema().load({"type": "report", "id": "abc-123"})
    assert "type" in excinfo.value.messages


def test_resource_schema_requires_id_and_type() -> None:
    """ResourceSchema must require both ``type`` and ``id``."""
    with pytest.raises(ValidationError) as excinfo:
        ResourceSchema().load({})
    assert "type" in excinfo.value.messages
    assert "id" in excinfo.value.messages


def test_rls_rule_schema_requires_clause() -> None:
    """RlsRuleSchema must require the ``clause`` field for RLS enforcement."""
    with pytest.raises(ValidationError) as excinfo:
        RlsRuleSchema().load({"dataset": 1})
    assert "clause" in excinfo.value.messages


def test_rls_rule_schema_accepts_full_payload() -> None:
    """RlsRuleSchema accepts a fully-specified rule with a dataset id."""
    loaded = RlsRuleSchema().load({"dataset": 7, "clause": "tenant_id = 1"})
    assert loaded == {"dataset": 7, "clause": "tenant_id = 1"}


def test_guest_token_create_schema_full_payload() -> None:
    """GuestTokenCreateSchema accepts a fully-formed embedded auth payload."""
    payload = {
        "user": {
            "username": "bob",
            "first_name": "Bob",
            "last_name": "Builder",
        },
        "resources": [{"type": "dashboard", "id": "uuid-1"}],
        "rls": [{"clause": "tenant_id = 1"}],
    }
    loaded = guest_token_create_schema.load(payload)
    assert loaded["user"]["username"] == "bob"
    assert loaded["resources"][0]["type"] == "dashboard"
    assert loaded["resources"][0]["id"] == "uuid-1"
    assert loaded["rls"][0]["clause"] == "tenant_id = 1"


def test_guest_token_create_schema_requires_resources_and_rls() -> None:
    """GuestTokenCreateSchema must require both ``resources`` and ``rls``."""
    with pytest.raises(ValidationError) as excinfo:
        GuestTokenCreateSchema().load({"user": {"username": "bob"}})
    assert "resources" in excinfo.value.messages
    assert "rls" in excinfo.value.messages


def test_role_response_schema_round_trip() -> None:
    """RoleResponseSchema dumps the public role projection used by /search."""
    schema = RoleResponseSchema()
    dumped = schema.dump(
        {
            "id": 1,
            "name": "Admin",
            "user_ids": [1, 2],
            "permission_ids": [10, 11],
        }
    )
    assert dumped == {
        "id": 1,
        "name": "Admin",
        "user_ids": [1, 2],
        "permission_ids": [10, 11],
    }


def test_roles_response_schema_round_trip() -> None:
    """RolesResponseSchema wraps a paginated list of roles for the API."""
    schema = RolesResponseSchema()
    dumped = schema.dump(
        {
            "count": 1,
            "ids": [1],
            "result": [
                {
                    "id": 1,
                    "name": "Admin",
                    "user_ids": [1],
                    "permission_ids": [2],
                }
            ],
        }
    )
    assert dumped["count"] == 1
    assert dumped["ids"] == [1]
    assert dumped["result"][0]["name"] == "Admin"


# ---------------------------------------------------------------------------
# CSRF endpoint unit test
# ---------------------------------------------------------------------------


def test_security_csrf_token_endpoint(
    client: Any,
    full_api_access: None,
) -> None:
    """
    Authenticated callers must receive a fresh CSRF token. This exercises the
    ``/api/v1/security/csrf_token/`` happy path that protects state-changing
    requests in the embedded/admin surfaces.
    """
    response = client.get("/api/v1/security/csrf_token/")
    assert response.status_code == 200
    assert response.json["result"] == generate_csrf()


# ---------------------------------------------------------------------------
# Guest token endpoint unit tests
# ---------------------------------------------------------------------------


def test_guest_token_returns_token_when_validator_passes(
    mocker: MockerFixture,
    app: Any,
    client: Any,
    full_api_access: None,
) -> None:
    """
    When the schema validates and the optional validator hook approves the
    payload, the endpoint returns a freshly minted guest token. This covers
    the success branch of ``SecurityRestApi.guest_token``.
    """
    appbuilder_mock = mocker.patch("superset.security.api.SecurityRestApi.appbuilder")
    appbuilder_mock.sm.validate_guest_token_resources.return_value = None
    appbuilder_mock.sm.create_guest_access_token.return_value = b"fake-token"

    payload = {
        "user": {"username": "bob", "first_name": "Bob", "last_name": "Builder"},
        "resources": [{"type": "dashboard", "id": "uuid-1"}],
        "rls": [{"clause": "tenant_id = 1"}],
    }
    with patch.dict(app.config, {"GUEST_TOKEN_VALIDATOR_HOOK": lambda body: True}):
        response = client.post("/api/v1/security/guest_token/", json=payload)

    assert response.status_code == 200
    assert response.json["token"] == "fake-token"
    appbuilder_mock.sm.validate_guest_token_resources.assert_called_once()
    appbuilder_mock.sm.create_guest_access_token.assert_called_once()


def test_guest_token_validator_rejects_payload(
    mocker: MockerFixture,
    app: Any,
    client: Any,
    full_api_access: None,
) -> None:
    """
    A validator hook that returns False must surface as a 400 response so
    callers learn the payload was rejected without a token being issued.
    """
    appbuilder_mock = mocker.patch("superset.security.api.SecurityRestApi.appbuilder")
    appbuilder_mock.sm.validate_guest_token_resources.return_value = None

    payload = {
        "user": {"username": "bob"},
        "resources": [{"type": "dashboard", "id": "uuid-1"}],
        "rls": [{"clause": "tenant_id = 1"}],
    }
    with patch.dict(app.config, {"GUEST_TOKEN_VALIDATOR_HOOK": lambda body: False}):
        response = client.post("/api/v1/security/guest_token/", json=payload)

    assert response.status_code == 400
    appbuilder_mock.sm.create_guest_access_token.assert_not_called()


def test_guest_token_validator_not_callable_raises_500(
    mocker: MockerFixture,
    app: Any,
    client: Any,
    full_api_access: None,
) -> None:
    """
    A misconfigured (non-callable) validator hook must raise the generic
    Superset error rather than silently issuing a token.
    """
    appbuilder_mock = mocker.patch("superset.security.api.SecurityRestApi.appbuilder")
    appbuilder_mock.sm.validate_guest_token_resources.return_value = None

    payload = {
        "user": {"username": "bob"},
        "resources": [{"type": "dashboard", "id": "uuid-1"}],
        "rls": [{"clause": "tenant_id = 1"}],
    }
    with patch.dict(app.config, {"GUEST_TOKEN_VALIDATOR_HOOK": "not-callable"}):
        with pytest.raises(SupersetGenericErrorException):
            client.post("/api/v1/security/guest_token/", json=payload)

    appbuilder_mock.sm.create_guest_access_token.assert_not_called()


def test_guest_token_returns_400_on_embedded_dashboard_not_found(
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
) -> None:
    """
    When the resource cannot be resolved to an embedded dashboard, the
    endpoint translates ``EmbeddedDashboardNotFoundError`` into a 400 response.
    """
    appbuilder_mock = mocker.patch("superset.security.api.SecurityRestApi.appbuilder")
    appbuilder_mock.sm.validate_guest_token_resources.side_effect = (
        EmbeddedDashboardNotFoundError()
    )

    payload = {
        "user": {"username": "bob"},
        "resources": [{"type": "dashboard", "id": "missing"}],
        "rls": [{"clause": "tenant_id = 1"}],
    }
    response = client.post("/api/v1/security/guest_token/", json=payload)
    assert response.status_code == 400


def test_guest_token_returns_400_on_invalid_payload(
    client: Any,
    full_api_access: None,
) -> None:
    """
    Schema-invalid payloads must return a 400 with ValidationError messages
    rather than reaching the token issuance path.
    """
    response = client.post("/api/v1/security/guest_token/", json={})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# RoleRestAPI.get_list unit tests
# ---------------------------------------------------------------------------


def test_role_search_rejects_invalid_order_column(
    client: Any,
    full_api_access: None,
) -> None:
    """
    Unsupported ``order_column`` values must return 400 instead of executing
    a query against an arbitrary column. This locks down the allowlist used
    in ``RoleRestAPI.get_list``.
    """
    response = client.get(
        "/api/v1/security/roles/search/?q=(order_column:not_a_column)"
    )
    assert response.status_code == 400
    assert "Invalid order column" in response.json["message"]


def _patch_role_query(mocker: MockerFixture) -> MagicMock:
    """Helper that patches ``db.session.query`` for RoleRestAPI tests."""
    query = MagicMock()
    query.options.return_value = query
    query.filter.return_value = query
    query.order_by.return_value = query
    query.offset.return_value = query
    query.limit.return_value = query
    query.count.return_value = 0
    query.all.return_value = []
    mocker.patch(
        "superset.security.api.db.session.query",
        return_value=query,
    )
    return query


@pytest.mark.parametrize("order_direction", ["asc", "desc"])
def test_role_search_supports_both_order_directions(
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
    order_direction: str,
) -> None:
    """
    Both ``asc`` and ``desc`` must be honored by ``RoleRestAPI.get_list``.
    """
    _patch_role_query(mocker)

    response = client.get(
        f"/api/v1/security/roles/search/?q=(order_direction:{order_direction})"
    )

    assert response.status_code == 200
    assert response.json == {"count": 0, "ids": [], "result": []}


@pytest.mark.parametrize(
    "filter_col,filter_value",
    [
        ("user_ids", 1),
        ("permission_ids", 2),
        ("group_ids", 3),
        ("name", "Admin"),
    ],
)
def test_role_search_applies_each_supported_filter(
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
    filter_col: str,
    filter_value: Any,
) -> None:
    """
    Each documented filter (``user_ids``, ``permission_ids``, ``group_ids``,
    ``name``) must be wired through to the query builder. This guards the
    filter dispatch table from silent regressions.
    """
    query = _patch_role_query(mocker)
    serialized_value = (
        f"'{filter_value}'" if isinstance(filter_value, str) else filter_value
    )

    response = client.get(
        "/api/v1/security/roles/search/"
        f"?q=(filters:!((col:{filter_col},value:{serialized_value})))"
    )

    assert response.status_code == 200, response.json
    assert query.filter.called


def test_role_search_translates_forbidden_error(
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
) -> None:
    """
    A ``ForbiddenError`` raised while building the query must be surfaced as
    a 403 instead of leaking details to the caller.
    """
    mocker.patch(
        "superset.security.api.db.session.query",
        side_effect=ForbiddenError("denied"),
    )
    response = client.get("/api/v1/security/roles/search/")
    assert response.status_code == 403


def test_role_search_translates_unexpected_error(
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
) -> None:
    """
    Any other unexpected exception in ``RoleRestAPI.get_list`` must be
    translated into a 500 response so transient failures are visible to
    operators.
    """
    mocker.patch(
        "superset.security.api.db.session.query",
        side_effect=RuntimeError("boom"),
    )
    response = client.get("/api/v1/security/roles/search/")
    assert response.status_code == 500


def test_role_search_serializes_role_payload(
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
) -> None:
    """
    The success path must project each role into the documented shape
    (``id``, ``name``, ``user_ids``, ``permission_ids``, ``group_ids``).
    """
    role = MagicMock()
    role.id = 1
    role.name = "Admin"
    user = MagicMock()
    user.id = 7
    permission = MagicMock()
    permission.id = 9
    group = MagicMock()
    group.id = 11
    role.user = [user]
    role.permissions = [permission]
    role.groups = [group]

    query = _patch_role_query(mocker)
    query.count.return_value = 1
    query.all.return_value = [role]

    response = client.get("/api/v1/security/roles/search/")

    assert response.status_code == 200
    body = response.json
    assert body["count"] == 1
    assert body["ids"] == [1]
    assert body["result"] == [
        {
            "id": 1,
            "name": "Admin",
            "user_ids": [7],
            "permission_ids": [9],
            "group_ids": [11],
        }
    ]
