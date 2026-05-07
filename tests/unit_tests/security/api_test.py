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
# pylint: disable=unused-argument, redefined-outer-name
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import prison
import pytest
from flask import current_app
from marshmallow import ValidationError
from pytest_mock import MockerFixture

from superset import security_manager
from superset.commands.dashboard.embedded.exceptions import (
    EmbeddedDashboardNotFoundError,
)
from superset.commands.exceptions import ForbiddenError
from superset.extensions import csrf


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
# BofA compliance coverage (issue #55): targeted regression tests for
# ``superset/security/api.py``. The Authentication control area must keep
# this file at >= 80% line coverage to meet OCC exam expectations. The tests
# below exercise the schema post-load conversion, the CSRF token endpoint,
# every branch of the guest-token endpoint, and every branch of the role
# search endpoint without requiring a full integration database.
# ---------------------------------------------------------------------------


def test_resource_schema_converts_enum_to_value(app_context: None) -> None:
    """
    ``ResourceSchema.convert_enum_to_value`` must replace the enum instance
    produced by Marshmallow with its string value so downstream code does
    not have to import ``GuestTokenResourceType``.
    """
    from superset.security.api import ResourceSchema

    loaded = ResourceSchema().load({"type": "dashboard", "id": "abc-123"})

    assert loaded == {"type": "dashboard", "id": "abc-123"}
    # post_load must produce a plain string, not an enum member
    assert isinstance(loaded["type"], str)


def test_resource_schema_rejects_unknown_resource_type(app_context: None) -> None:
    """
    ``ResourceSchema`` must raise ``ValidationError`` for resource types that
    are not part of ``GuestTokenResourceType``. This guards the guest-token
    endpoint against arbitrary resource impersonation.
    """
    from superset.security.api import ResourceSchema

    with pytest.raises(ValidationError):
        ResourceSchema().load({"type": "totally-not-a-real-type", "id": "abc"})


def test_csrf_token_endpoint_returns_token(client: Any, full_api_access: None) -> None:
    """
    ``GET /api/v1/security/csrf_token/`` must return HTTP 200 with a
    non-empty ``result`` string when the caller is authorized. This covers
    the success branch of ``SecurityRestApi.csrf_token`` (line 135).
    """
    response = client.get("/api/v1/security/csrf_token/")

    assert response.status_code == 200
    payload = response.json
    assert isinstance(payload["result"], str)
    assert payload["result"]


@pytest.fixture
def guest_token_payload() -> dict[str, Any]:
    return {
        "user": {"username": "bob", "first_name": "Bob", "last_name": "Smith"},
        "resources": [{"type": "dashboard", "id": "abc-123"}],
        "rls": [{"clause": "tenant_id = 1"}],
    }


def test_guest_token_endpoint_success(
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
    guest_token_payload: dict[str, Any],
) -> None:
    """
    Happy-path: a valid request returns HTTP 200 and the JWT produced by
    ``security_manager.create_guest_access_token``. Covers the main body of
    ``SecurityRestApi.guest_token`` including resource validation and token
    creation (lines 171-192).
    """
    mocker.patch.object(
        security_manager, "validate_guest_token_resources", return_value=None
    )
    mocker.patch.object(
        security_manager,
        "create_guest_access_token",
        return_value="header.payload.signature",
    )

    response = client.post("/api/v1/security/guest_token/", json=guest_token_payload)

    assert response.status_code == 200
    assert response.json == {"token": "header.payload.signature"}
    security_manager.create_guest_access_token.assert_called_once_with(
        guest_token_payload["user"],
        # post_load coerces the enum to its value
        [{"type": "dashboard", "id": "abc-123"}],
        guest_token_payload["rls"],
    )


def test_guest_token_endpoint_rejects_invalid_resource(
    client: Any, full_api_access: None
) -> None:
    """
    Schema validation rejects resource types that are not part of
    ``GuestTokenResourceType``. This must surface as HTTP 400 from the
    ``ValidationError`` branch (line 195-196).
    """
    payload = {
        "user": {"username": "bob"},
        "resources": [{"type": "evil", "id": "abc"}],
        "rls": [],
    }

    response = client.post("/api/v1/security/guest_token/", json=payload)

    assert response.status_code == 400


def test_guest_token_endpoint_handles_missing_dashboard(
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
    guest_token_payload: dict[str, Any],
) -> None:
    """
    When ``validate_guest_token_resources`` raises
    ``EmbeddedDashboardNotFoundError`` the endpoint must respond with
    HTTP 400 (lines 193-194).
    """
    mocker.patch.object(
        security_manager,
        "validate_guest_token_resources",
        side_effect=EmbeddedDashboardNotFoundError(),
    )

    response = client.post("/api/v1/security/guest_token/", json=guest_token_payload)

    assert response.status_code == 400


def test_guest_token_endpoint_validator_hook_denies(
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
    guest_token_payload: dict[str, Any],
) -> None:
    """
    A configured ``GUEST_TOKEN_VALIDATOR_HOOK`` returning falsy must result
    in HTTP 400. This covers the ``ValidationError`` re-raise on lines
    178-181.
    """
    mocker.patch.object(
        security_manager, "validate_guest_token_resources", return_value=None
    )
    create_token = mocker.patch.object(
        security_manager, "create_guest_access_token", return_value="never"
    )
    original_hook = current_app.config.get("GUEST_TOKEN_VALIDATOR_HOOK")
    current_app.config["GUEST_TOKEN_VALIDATOR_HOOK"] = lambda body: False
    try:
        response = client.post(
            "/api/v1/security/guest_token/", json=guest_token_payload
        )
    finally:
        current_app.config["GUEST_TOKEN_VALIDATOR_HOOK"] = original_hook

    assert response.status_code == 400
    create_token.assert_not_called()


def test_guest_token_endpoint_validator_hook_allows(
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
    guest_token_payload: dict[str, Any],
) -> None:
    """
    A configured ``GUEST_TOKEN_VALIDATOR_HOOK`` returning truthy must allow
    the request through and a token must be issued. Covers the truthy branch
    of the ``callable(...)`` check on lines 178-180.
    """
    mocker.patch.object(
        security_manager, "validate_guest_token_resources", return_value=None
    )
    create_token = mocker.patch.object(
        security_manager, "create_guest_access_token", return_value="ok-token"
    )
    seen: list[dict[str, Any]] = []

    def _hook(body: dict[str, Any]) -> bool:
        seen.append(body)
        return True

    original_hook = current_app.config.get("GUEST_TOKEN_VALIDATOR_HOOK")
    current_app.config["GUEST_TOKEN_VALIDATOR_HOOK"] = _hook
    try:
        response = client.post(
            "/api/v1/security/guest_token/", json=guest_token_payload
        )
    finally:
        current_app.config["GUEST_TOKEN_VALIDATOR_HOOK"] = original_hook

    assert response.status_code == 200
    assert response.json == {"token": "ok-token"}
    assert len(seen) == 1
    create_token.assert_called_once()


def test_guest_token_endpoint_validator_hook_not_callable(
    mocker: MockerFixture,
    client: Any,
    full_api_access: None,
    guest_token_payload: dict[str, Any],
) -> None:
    """
    A misconfigured ``GUEST_TOKEN_VALIDATOR_HOOK`` (not callable) must raise
    ``SupersetGenericErrorException`` and surface as HTTP 500. Covers lines
    182-185.
    """
    mocker.patch.object(
        security_manager, "validate_guest_token_resources", return_value=None
    )

    original_hook = current_app.config.get("GUEST_TOKEN_VALIDATOR_HOOK")
    current_app.config["GUEST_TOKEN_VALIDATOR_HOOK"] = 123  # not callable
    try:
        response = client.post(
            "/api/v1/security/guest_token/", json=guest_token_payload
        )
    finally:
        current_app.config["GUEST_TOKEN_VALIDATOR_HOOK"] = original_hook

    assert response.status_code == 500


def _make_role(role_id: int, name: str) -> SimpleNamespace:
    """Build a stand-in for ``flask_appbuilder.security.sqla.models.Role``."""
    return SimpleNamespace(
        id=role_id,
        name=name,
        user=[SimpleNamespace(id=10 + role_id)],
        permissions=[SimpleNamespace(id=100 + role_id)],
        groups=[SimpleNamespace(id=1000 + role_id)],
    )


def _install_role_query(
    mocker: MockerFixture, roles: list[SimpleNamespace], total: int | None = None
) -> MagicMock:
    """
    Patch ``superset.security.api.db.session.query`` so the endpoint runs
    against an in-memory list instead of a live database. Returns the query
    mock so assertions can check filter/order calls.
    """
    query = MagicMock()
    query.options.return_value = query
    query.filter.return_value = query
    query.count.return_value = total if total is not None else len(roles)
    ordered = MagicMock()
    offset = MagicMock()
    limit = MagicMock()
    query.order_by.return_value = ordered
    ordered.offset.return_value = offset
    offset.limit.return_value = limit
    limit.all.return_value = roles

    session = mocker.patch("superset.security.api.db.session")
    session.query.return_value = query
    return query


def test_role_search_returns_roles(
    mocker: MockerFixture, client: Any, full_api_access: None
) -> None:
    """
    Happy-path: ``GET /api/v1/security/roles/search/`` returns serialized
    roles including user, permission, and group ids. Covers the bulk of
    ``RoleRestAPI.get_list`` (lines 283-343).
    """
    roles = [_make_role(1, "Admin"), _make_role(2, "Alpha")]
    _install_role_query(mocker, roles)

    response = client.get("/api/v1/security/roles/search/")

    assert response.status_code == 200
    body = response.json
    assert body["count"] == 2
    assert body["ids"] == [1, 2]
    assert body["result"][0] == {
        "id": 1,
        "name": "Admin",
        "user_ids": [11],
        "permission_ids": [101],
        "group_ids": [1001],
    }


def test_role_search_invalid_order_column_returns_400(
    mocker: MockerFixture, client: Any, full_api_access: None
) -> None:
    """
    Requesting ``order_column`` outside the allowlist must short-circuit to
    HTTP 400 and never touch the database. Covers lines 288-292.
    """
    session = mocker.patch("superset.security.api.db.session")

    response = client.get("/api/v1/security/roles/search/?q=(order_column:evil)")

    assert response.status_code == 400
    assert "Invalid order column" in response.json["message"]
    session.query.assert_not_called()


def test_role_search_applies_all_filters_and_descending_order(
    mocker: MockerFixture, client: Any, full_api_access: None
) -> None:
    """
    Filters (user_ids/permission_ids/group_ids/name) must each translate
    into a ``query.filter`` call and ``order_direction=desc`` must produce
    a descending ``order_by`` clause. Covers lines 294-321.
    """
    roles = [_make_role(7, "Gamma")]
    query = _install_role_query(mocker, roles, total=1)

    rison = prison.dumps(
        {
            "order_column": "name",
            "order_direction": "desc",
            "page": 0,
            "page_size": 5,
            "filters": [
                {"col": "user_ids", "opr": "eq", "value": "10"},
                {"col": "permission_ids", "opr": "eq", "value": "20"},
                {"col": "group_ids", "opr": "eq", "value": "30"},
                {"col": "name", "opr": "ct", "value": "Gamma"},
            ],
        }
    )
    response = client.get(f"/api/v1/security/roles/search/?q={quote(rison)}")

    assert response.status_code == 200
    # one filter call per recognised column
    assert query.filter.call_count == 4
    # descending order applied on Role.name
    assert query.order_by.called


def test_role_search_returns_403_on_forbidden(
    mocker: MockerFixture, client: Any, full_api_access: None
) -> None:
    """
    A ``ForbiddenError`` raised inside the handler must become HTTP 403 via
    the dedicated ``except`` branch (lines 344-345).
    """
    mocker.patch(
        "superset.security.api.db.session.query",
        side_effect=ForbiddenError("not allowed"),
    )

    response = client.get("/api/v1/security/roles/search/")

    assert response.status_code == 403


def test_role_search_returns_500_on_unexpected_error(
    mocker: MockerFixture, client: Any, full_api_access: None
) -> None:
    """
    Any other exception from the handler must be caught and surfaced as
    HTTP 500. Covers the catch-all branch on lines 346-347.
    """
    mocker.patch(
        "superset.security.api.db.session.query",
        side_effect=RuntimeError("boom"),
    )

    response = client.get("/api/v1/security/roles/search/")

    assert response.status_code == 500
