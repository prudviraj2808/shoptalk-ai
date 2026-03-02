# ShopTalk AI Data Preparation & Finetuning

This guide describes the complete workflow for preparing your dataset and finetuning the MobileCLIP model with LoRA adapters.

---

## 1. Prepare Training Data

Extract, balance, and organize your dataset using:

```bash
uv run train/prepare_training_data.py
```

- Extracts metadata and images from tar files and CSV.
- Balances the dataset so each product category has exactly 100 entries (upsampling or downsampling as needed).
- Saves images to `training_data/images/` in subfolders by hash prefix.
- Outputs a JSONL file `training_data/full_metadata.jsonl` with records like:
  - `item_id`: unique item identifier
  - `text`: item name or caption
  - `image_path`: relative path to the extracted image
  - `metadata`: additional attributes

---

## 2. Metadata Refinement

Refine the metadata for training using:

```bash
uv run train/sagemaker_metadata_refinement.py
```

- Processes `full_metadata.jsonl` to produce `refined_sagemaker_metadata.jsonl`.
- Typical output fields:
  - `image_path`: relative path to the image
  - `refined_caption`: improved or cleaned text caption

---

## 3. Convert to WebDataset Format

Package images and captions into tar files for efficient training:

```bash
uv run train/conversion_script.py
```

- Creates `mobileclip_data_*.tar` files in the project root.
- Each tar contains JPEG images and captions with keys `jpg` and `txt`.

---

## 4. Finetune MobileCLIP with LoRA

Edit environment variables as needed (model name, batch size, etc.), then run:

```bash
uv run train/finetune.py
```

- Loads tar files, applies LoRA adapters, and trains the model.
- LoRA weights are saved to `output/mobileclip2_lora` (or as configured).

---

## Notes

- Make sure all required Python packages are installed (see `requirements.txt`).
- For Windows, ensure the entry point in `finetune.py` is protected with `if __name__ == "__main__":`.
- Adjust paths and parameters as needed for your environment.
- Each script prints progress and warnings for missing or problematic files.

---

## Example Data Flow

1. `prepare_training_data.py` → `training_data/full_metadata.jsonl` + images
2. `sagemaker_metadata_refinement.py` → `training_data/refined_sagemaker_metadata.jsonl`
3. `conversion_script.py` → `mobileclip_data_*.tar`
4. `finetune.py` → LoRA weights in `output/`

---

## Troubleshooting

- Warnings about missing images: Check your image paths and ensure all files exist.
- Errors about input shapes: Ensure batching is handled only once in the pipeline.
- Multiprocessing issues: Use top-level functions and protect entry points for Windows compatibility.

---

## License

See LICENSE-CC-BY-4.0.txt in the data folders for image licensing information.

---

For questions or help, contact the project maintainer.


## Localhost Endpoints & Services

Your Docker Compose setup launches three main services, each with its own role and endpoint:

### 1. Database (PostgreSQL with pgvector)
- **Service name:** db
- **Port mapping:** 5432:5432
- **Purpose:** Stores all application data and embeddings. Both app and adk-ui depend on this service.
- **Connection details:**
  - Host (from your machine): localhost
  - Port: 5432
  - User: user
  - Password: pass
  - Database: shoptalk
  - Connection string (local): `postgresql://user:pass@localhost:5432/shoptalk`
  - Connection string (internal): `postgresql://user:pass@db:5432/shoptalk`
  - Use `db` as the host internally because Docker Compose creates a network and uses service names as hostnames.


### 2. ADK Development UI
- **Service name:** adk-ui
- **URL:** http://localhost:5000
- **Purpose:** Provides the ADK web interface, limited to agent selection in the agents/ folder.
- **Depends on:** db (waits for database to be healthy before starting)

> The db service is listed first in docker-compose.yml because both app and adk-ui require the database to be available and healthy before they start. This ensures reliable startup and avoids connection errors.

---

## Project Folder Structure

```
shoptalk-ai/
├── .env
├── README.md
├── docker-compose.yml
├── pyproject.toml
├── data/
│   ├── images.csv
│   ├── abo-images-small/
│   └── abo-listings/
├── docker/
│   └── Dockerfile
├── models/
├── agents/
│   ├── agent.py
├── tools/
│   ├── product_search.py
├── train/
│   ├── dataset.py
│   ├── finetune.py
│   ├── prepare_training_data.py
│   └── sagemaker_metadata_refinement.py
├── training_data/
│   ├── full_metadata.jsonl
│   ├── refined_sagemaker_metadata.jsonl
│   ├── README.md
│   └── images/
│       ├── 00/
│       ├── 01/
│       └── ...
```

> This structure highlights the main directories and files. Some folders (like images/) may contain many subfolders/files not shown here for brevity.