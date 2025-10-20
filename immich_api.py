from datetime import date
import requests
import logging

logger = logging.getLogger(__name__)

# needed permissions:
# - server.about (ping the server)
# - asset.read (list assets)
# - asset.download (download original image)
# - person.read (list people, get birth date)


def validate_immich_connection(api_key, base_url):
    """
    Validates that the provided Immich API key and base URL are working.

    Args:
        api_key (str): API key for authentication.
        base_url (str): Base URL of the API.

    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    if not api_key or not base_url:
        return False, "API key and base URL are required."

    try:
        headers = {
            'Accept': 'application/json',
            'x-api-key': api_key,
        }
        # Try a simple ping to the server via the user endpoint
        url = f"{base_url}/server/about"
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            return True, "Connection successful."
        elif response.status_code == 401:
            return False, "Authentication failed. Invalid API key."
        else:
            return False, f"Server error: Status code {response.status_code}"

    except requests.exceptions.ConnectionError:
        return False, "Connection error. Check the base URL."
    except requests.exceptions.Timeout:
        return False, "Connection timed out. Server might be down."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def get_assets_with_person(api_key, base_url, person_id, date_from=None, date_to=None):
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
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-api-key': api_key,
    }
    url = f"{base_url}/search/metadata"
    all_assets = []
    payload = {
        "page": 1,
        "type": "IMAGE",
        "personIds": [person_id],
        "visibility": "timeline",
        "withDeleted": False,
        "withExif": True,
        "withPeople": True,
        "withStacked": True,
    }

    if date_from:
        payload["takenAfter"] = f"{date_from}T00:00:00.000Z"

    if date_to:
        payload["takenBefore"] = f"{date_to}T23:59:59.999Z"

    while payload["page"] is not None:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            logger.info(f"Error fetching page {payload['page']}: {response.status_code} - {response.text}")
            break
        data = response.json()
        if not data:
            break
        all_assets.extend(data['assets']['items'])
        logger.info(f"Fetched page {payload['page']} with {len(data['assets']['items'])} assets")
        payload["page"] = data['assets'].get('nextPage')
    return all_assets


def get_birth_date(api_key, base_url, person_id):
    """
    Retrieves the birth date of a person from the API.

    Args:
        api_key (str): API key for authentication.
        base_url (str): Base URL of the API.
        person_id (str): ID of the person to search for.

    Returns:
        date: Birth date of the person if found, None otherwise.
    """
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'x-api-key': api_key,
    }
    url = f"{base_url}/people/{person_id}"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.warn(f"Error fetching birth date: {response.status_code} - {response.text}")
        return None
    data = response.json()
    if not data:
        logger.warn(f"Error fetching birth date: no body in response")
        return None
    birth_date_str = data['birthDate']
    if not birth_date_str:
        return None
    return date.fromisoformat(birth_date_str)


def download_asset(api_key, base_url, asset_id):
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
