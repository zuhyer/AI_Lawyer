import requests
from pathlib import Path

from AI_Lawyer.entity.config_entity import DataConfig
from AI_Lawyer.utils.logging_setup import logger

class DataIngestion:
    def __init__(self, config: DataConfig):
        self.config = config
        self.pdf_dir = Path(self.config.pdf_directory)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def download_pdfs(self):
        for url in self.config.source_url:
            try:
                file_name = url.split("/")[-1]
                if "?" in file_name:
                    file_name = file_name.split("?")[0]  # Clean filename if there are query params

                save_path = self.pdf_dir / file_name

                # Check if already downloaded
                if save_path.exists():
                    logger.info(f"Already downloaded. Skipping: {save_path}")
                    continue

                response = requests.get(url, timeout=10)
                response.raise_for_status()  # Raises HTTPError for bad responses

                with open(save_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"Successfully downloaded: {url} -> {save_path}")

            except requests.exceptions.RequestException as e:
                logger.error(f"FAILED to download {url}. Error: {e}")

    def main(self):
        self.download_pdfs()

