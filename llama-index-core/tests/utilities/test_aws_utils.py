import pytest
from unittest.mock import MagicMock
import aioboto3
from llama_index.core.utilities.aws_utils import aget_aws_service_session


@pytest.fixture()
def mock_aioboto3():
    mock_session = MagicMock(spec=aioboto3.Session)
    mock_client = MagicMock(spec=aioboto3.Session.client)
    mock_session.client.return_value = mock_client
    return mock_session


def test_aget_aws_service_session(mock_aioboto3):
    # Call the function with the necessary arguments
    session = aget_aws_service_session(
        region_name="us-west-2",
        aws_access_key_id="access_key",
        aws_secret_access_key="secret_key",
        aws_session_token="session_token",
        profile_name="default",
    )

    # Assert that the returned client is of the correct type
    assert isinstance(session, aioboto3.Session)
