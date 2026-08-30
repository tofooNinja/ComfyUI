import os

import pytest

import folder_paths
from app.model_download import ModelDownloadError, validate_download_request

HF_URL = "https://huggingface.co/org/repo/resolve/main/model.safetensors"


def test_valid_request_resolves_into_folder():
    dest = validate_download_request(HF_URL, "vae", "model.safetensors")
    assert os.path.dirname(dest) == os.path.abspath(folder_paths.get_folder_paths("vae")[0])
    assert os.path.basename(dest) == "model.safetensors"


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/org/repo/resolve/main/model.safetensors",
        "https://example.com/model.safetensors",
        "https://huggingface.co.evil.com/model.safetensors",
        "ftp://huggingface.co/model.safetensors",
        "not a url",
    ],
)
def test_rejects_untrusted_urls(url):
    with pytest.raises(ModelDownloadError):
        validate_download_request(url, "vae", "model.safetensors")


@pytest.mark.parametrize("folder", ["custom_nodes", "configs", "diffusers", "does_not_exist", "../vae"])
def test_rejects_bad_folders(folder):
    with pytest.raises(ModelDownloadError):
        validate_download_request(HF_URL, folder, "model.safetensors")


@pytest.mark.parametrize(
    "filename",
    ["../model.safetensors", "sub/model.safetensors", "..", ".hidden.safetensors", "", "model.py", "a\\b.safetensors"],
)
def test_rejects_bad_filenames(filename):
    with pytest.raises(ModelDownloadError):
        validate_download_request(HF_URL, "vae", filename)
