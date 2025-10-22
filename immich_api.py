
from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID
import requests
import logging
from immich_client import AuthenticatedClient

from immich_client.api.search import search_assets
from immich_client.models.asset_response_dto import AssetResponseDto
from immich_client.models.asset_type_enum import AssetTypeEnum
from immich_client.models.asset_visibility import AssetVisibility
from immich_client.models.metadata_search_dto import MetadataSearchDto

logger = logging.getLogger(__name__)

# needed permissions:
# - server.about (ping the server)
# - asset.read (list assets)
# - asset.download (download original image)


@dataclass(frozen=True)
class ImmichConnectionInfo:
    is_success: bool
    error_message: str | None

def validate_immich_connection(api_key: str, base_url: str) -> ImmichConnectionInfo:
    """
    Validates that the provided Immich API key and base URL are working.

    Args:
        api_key (str): API key for authentication.
        base_url (str): Base URL of the API.

    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    if not api_key or not base_url:
        return ImmichConnectionInfo(is_success=False, error_message="API key and base URL are required.")

    try:
        headers = {
            'Accept': 'application/json',
            'x-api-key': api_key,
        }
        # Try a simple ping to the server via the user endpoint
        url = f"{base_url}/server/about"
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            return ImmichConnectionInfo(is_success=True, error_message=None)
        elif response.status_code == 401:
            return ImmichConnectionInfo(is_success=False, error_message="Authentication failed. Invalid API key.")
        else:
            return ImmichConnectionInfo(is_success=False, error_message=f"Server error: Status code {response.status_code}")

    except requests.exceptions.ConnectionError:
        return ImmichConnectionInfo(is_success=False, error_message="Connection error. Check the base URL.")
    except requests.exceptions.Timeout:
        return ImmichConnectionInfo(is_success=False, error_message="Connection timed out. Server might be down.")
    except Exception as e:
        return ImmichConnectionInfo(is_success=False, error_message=f"Unexpected error: {str(e)}")
 


def get_assets_with_person(api_key: str, base_url: str, person_id: UUID, date_from: date | None, date_to: date | None) -> list[AssetResponseDto]:
    """
    Retrieve all image assets containing the specified person by querying the API.

    Args:
        api_key (str): API key for authentication.
        base_url (str): Base URL of the API.
        person_id (str): ID of the person to search for.
        date_from (str, optional): Start date in ISO format (YYYY-MM-DD).
        date_to (str, optional): End date in ISO format (YYYY-MM-DD).

    Returns:
        list: List of asset dictionaries.
    """
    payload = MetadataSearchDto(
        page=1,
        type_=AssetTypeEnum.IMAGE,
        person_ids=[person_id],
        visibility=AssetVisibility.TIMELINE,
        with_deleted=False,
        with_exif=True,
        with_people=True,
        with_stacked=True,
    )

    if date_from:
        payload.taken_after = datetime.combine(date_from, datetime.min.time())

    if date_to:
        payload.taken_before = datetime.combine(date_to, datetime.max.time())

    client = AuthenticatedClient(base_url=base_url, token=api_key)
    

    all_assets: list[AssetResponseDto] = []
    while payload.page:
        response = search_assets.sync_detailed(client=client, body=payload)
        if response.status_code != 200:
            logger.info(f"Error fetching page {payload.page}: {response.status_code} - {str(response.content)}")
            break
        if not response.parsed:
            break
        all_assets.extend(response.parsed.assets.items)
        logger.info(f"Fetched page {payload.page} with {len(response.parsed['assets']['items'])} assets")
        payload.page = response.parsed['assets'].get('nextPage')
    return all_assets

def download_asset(api_key: str, base_url: str, asset_id: str) -> bytes:
    """
    Downloads the original image asset from the API.

    Args:
        api_key (str): API key for authentication.
        base_url (str): Base URL of the API.
        asset_id (str): The asset's ID.

    Returns:
        bytes: The content of the downloaded image.
    """
    headers = {'x-api-key': api_key}
    response = requests.get(f'{base_url}/assets/{asset_id}/original', headers=headers)
    response.raise_for_status()
    return response.content
