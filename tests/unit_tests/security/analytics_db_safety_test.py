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

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.engine.url import make_url

from superset.exceptions import SupersetSecurityException
from superset.security.analytics_db_safety import (
    check_sqlalchemy_uri,
    get_sqlalchemy_uri_blocklist,
)


@patch(
    "superset.security.analytics_db_safety.feature_flag_manager.is_feature_enabled",
    return_value=False,
)
def test_superset_meta_db_uri_blocked_when_feature_disabled(
    mock_is_feature_enabled: MagicMock,
) -> None:
    with pytest.raises(SupersetSecurityException, match="cannot be used"):
        check_sqlalchemy_uri(make_url("superset://"))

    mock_is_feature_enabled.assert_called_once_with("ENABLE_SUPERSET_META_DB")


@patch(
    "superset.security.analytics_db_safety.feature_flag_manager.is_feature_enabled",
    return_value=True,
)
def test_superset_meta_db_uri_allowed_when_feature_enabled(
    mock_is_feature_enabled: MagicMock,
) -> None:
    check_sqlalchemy_uri(make_url("superset://"))

    mock_is_feature_enabled.assert_called_once_with("ENABLE_SUPERSET_META_DB")


@patch("superset.security.analytics_db_safety.feature_flag_manager.is_feature_enabled")
def test_superset_meta_db_blocklist_does_not_stick_across_feature_flag_changes(
    mock_is_feature_enabled: MagicMock,
) -> None:
    mock_is_feature_enabled.side_effect = [False, True]

    disabled_blocklist = get_sqlalchemy_uri_blocklist()
    enabled_blocklist = get_sqlalchemy_uri_blocklist()

    assert any(pattern.pattern == "superset$" for pattern in disabled_blocklist)
    assert all(pattern.pattern != "superset$" for pattern in enabled_blocklist)


@pytest.mark.parametrize(
    "sqlalchemy_uri, error",
    [
        ("postgres://user:password@test.com", False),
        (
            "sqlite:///home/superset/bad.db",
            True,
        ),
        (
            "sqlite+pysqlite:///home/superset/bad.db",
            True,
        ),
        (
            "sqlite+aiosqlite:///home/superset/bad.db",
            True,
        ),
        (
            "sqlite+pysqlcipher:///home/superset/bad.db",
            True,
        ),
        (
            "sqlite+:///home/superset/bad.db",
            True,
        ),
        (
            "sqlite+new+driver:///home/superset/bad.db",
            True,
        ),
        (
            "sqlite+new+:///home/superset/bad.db",
            True,
        ),
        (
            "shillelagh:///home/superset/bad.db",
            True,
        ),
        (
            "shillelagh+apsw:///home/superset/bad.db",
            True,
        ),
        ("shillelagh+:///home/superset/bad.db", False),
        (
            "shillelagh+something:///home/superset/bad.db",
            False,
        ),
    ],
)
@patch(
    "superset.security.analytics_db_safety.feature_flag_manager.is_feature_enabled",
    return_value=False,
)
def test_check_sqlalchemy_uri(
    mock_is_feature_enabled: MagicMock,
    sqlalchemy_uri: str,
    error: bool,
) -> None:
    if error:
        with pytest.raises(SupersetSecurityException) as excinfo:
            check_sqlalchemy_uri(make_url(sqlalchemy_uri))

        assert "cannot be used as a data source for security reasons" in str(
            excinfo.value
        )
    else:
        check_sqlalchemy_uri(make_url(sqlalchemy_uri))

    mock_is_feature_enabled.assert_called_with("ENABLE_SUPERSET_META_DB")
