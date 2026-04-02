import hydra
from omegaconf import DictConfig
from SDAT.core import run_vqvae_training


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_vqvae_training(cfg)


if __name__ == "__main__":
    main()
