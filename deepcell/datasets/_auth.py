"""User interface to authentication layer for data/models."""

import os
import requests
from pathlib import Path
from hashlib import md5
from tqdm import tqdm
import logging


_api_endpoint = "https://users.deepcell.org/api/getData/"
_asset_location = Path.home() / ".deepcell"


# TODO s:
#  - Add data caching + force option
#  - Make download location a kwarg?
def fetch_data(asset_key: str, cache_subdir=None, file_hash=None) -> None:
    """Fetch assets through deepcell-connect authentication system.

    Download assets from the deepcell suite of datasets and models which
    require user-authentication.

    .. note::

       You must have a Deepcell Access Token set as an environment variable
       with the name ``DEEPCELL_ACCESS_TOKEN`` in order to access assets.

       Access tokens can be created at <https://users.deepcell.org>_

    Args:
        :param asset_key: Key of the file to download.
        The list of available assets can be found on the deepcell-connect 
        homepage.
    """
    logging.basicConfig(level=logging.INFO)

    download_location = _asset_location
    if cache_subdir is not None:
        download_location /= cache_subdir
    download_location.mkdir(exist_ok=True)

    # Extract the filename from the asset_key, which can be a full path
    fname = os.path.split(asset_key)[-1]

    # Check for cached data
    if file_hash is not None:
        with open(fname, "rb") as fh:
            hasher = md5(fh.read())
        logging.info(f"Checking {fname} against provided file_hash...")
        md5sum = hasher.hexdigest()
        if md5sum == file_hash:
            logging.info(f"{fname} with hash {file_hash} already available.")
            return
        logging.info(f"{fname} with hash {file_hash} not found in {download_location}")

    # Check for access token
    access_token = os.environ.get("DEEPCELL_ACCESS_TOKEN")
    if access_token is None:
        raise ValueError(
            (
                "DEEPCELL_ACCESS_TOKEN not found.\n",
                "Please set your access token to the DEEPCELL_ACCESS_TOKEN\n",
                "environment variable.\n",
                "For example:\n\n",
                "\texport DEEPCELL_ACCESS_TOKEN=<your-token>.\n\n",
                "If you don't have a token, create one at deepcell-connect.",
            )
        )

    # Request download URL
    headers = {"X-Api-Key": access_token}
    logging.info("Making request to server")
    resp = requests.post(
        _api_endpoint, headers=headers, data={"s3_key": asset_key}
    )
    resp.raise_for_status()  # Raise exception if not 200 status

    # Parse response
    response_data = resp.json()
    download_url = response_data["url"]
    file_size = response_data["size"]
    # Parse file_size (TODO: would be more convenient if it were numerical, i.e. always bytes)
    val, suff = file_size.split(" ")
    # TODO: Case statement would be awesome here, but need to support all the
    # way back to Python 3.8
    suffix_mapping = {"kB": 1e3, "MB": 1e6, "B": 1, "GB": 1e9}
    file_size_numerical = int(float(val) * suffix_mapping[suff])

    logging.info(
        f"Downloading {asset_key} with size {file_size} to {download_location}"
    )
    data_req = requests.get(
        download_url, headers={"user-agent": "Wget/1.20 (linux-gnu)"}, stream=True
    )
    data_req.raise_for_status()

    chunk_size = 4096
    fpath = download_location / fname
    with tqdm.wrapattr(
        open(fpath, "wb"), "write", miniters=1, total=file_size_numerical
    ) as fh:
        for chunk in data_req.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
    
    logging.info(f"🎉 Successfully downloaded file to {fpath}")
